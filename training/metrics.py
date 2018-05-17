"""
=================
This file contains all the metrics that is used for evaluation. 
=================
"""

import torch
from pyquaternion import Quaternion
import numpy as np
import math
import pdb
import torch.nn as nn
import matplotlib.pyplot as plt


class BaseMetric:

    def record_output(self, output, output_indices, target, prev_absolutes,
                      next_absolutes, batch_size=1):
        raise 'record_output is not implemented'

    def report(self):
        raise 'report is not implemented'

    def final_report(self):
        raise 'final_report is not implemented'


class SequenceMultiClassMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()
        self.confusion = None

    def record_output(self, output, output_indices, target, prev_absolutes,
                      next_absolutes, batch_size=1):
        assert output.dim() == 4
        assert target.dim() == 3

        _, predictions = output.max(3)

        # Compute per class accuracy for unbalanced data.
        sequence_length = output.size(1)
        num_label = output.size(2)
        num_class = output.size(3)

        if self.confusion is None:
            # Confusion matrix is 4D because it's defined per label and sequence
            # element.
            self.confusion = torch.zeros(sequence_length, num_label, num_class,
                                         num_class)
        # Compute per class accuracy in this batch and update the confusion
        # matrix.
        per_class_acc = []
        for seq_id in range(sequence_length):
            for imu_id in range(num_label):
                imu_target = target[:, seq_id, imu_id].contiguous()
                imu_preds = predictions[:, seq_id, imu_id].contiguous()
                for label, pred in zip(imu_target.view(-1), imu_preds.view(-1)):
                    self.confusion[seq_id, imu_id, label, pred] += 1.0
                for class_id in range(num_class):
                    # Look at targets where label is class_id, and see what
                    # percentage of predictions are class_id.
                    preds_for_class = imu_preds[imu_target == class_id]
                    if len(preds_for_class) > 0:
                        per_class_acc.append(
                            (preds_for_class == class_id).float().mean())
        per_class_acc = sum(per_class_acc) / len(per_class_acc)
        accuracy = (predictions == target).float().mean()
        self.meter.update(
            torch.Tensor([100 * accuracy, 100 * per_class_acc]), batch_size)

    def report(self):
        return ('Accuracy {meter.val[0]:.2f} ({meter.avg[0]:.2f})    ' +
                'Balanced {meter.val[1]:.2f} ({meter.avg[1]:.2f})'
               ).format(meter=self.meter)

    def final_report(self):
        correct_preds = self.confusion[:, :,
                                       range(self.args.num_classes),
                                       range(self.args.num_classes)]
        correct_percentage = correct_preds / (self.confusion.sum(3) + 1e-6) * 100
        balance_accuracy = correct_percentage.mean()
        per_sequence_element_accuracy = correct_percentage.view(
            correct_percentage.size(0), -1).mean(1)
        per_sequence_report = ', '.join(
            '{:.2f}'.format(acc) for acc in per_sequence_element_accuracy)
        report = ('Accuracy {meter.avg[0]:.2f}   Balanced {balanced:.2f}   '
                  'PerSeq [{per_seq}]').format(meter=self.meter,
                                               balanced=balance_accuracy,
                                               per_seq=per_sequence_report)
        report += '   Accuracy Matrix (seq x imu x label): {}'.format(
            correct_percentage)
        return report


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0


class AngleClassificationMetric(BaseMetric):

    def __init__(self, args):
        self.centroids = args.relative_centroids
        self.metric = SequenceMultiClassMetric(args)

    def record_output(self, output, output_indices, target, prev_absolute_imu,
                      next_absolute_imu, batch_size=1):
        # return
        assert output.dim() == 4
        size = output.size()
        output_labels = torch.LongTensor(size[0], size[1], size[2],
                                         self.centroids.size(1)).zero_()
        target_labels = torch.LongTensor(size[0], size[1], size[2])
        for batch_id in range(size[0]):
            for seq_id in range(size[1]):
                for imu_id in range(size[2]):
                    output_distances = []
                    target_distances = []
                    for centroid in self.centroids[imu_id]:
                        output_distances.append(
                            self.get_angle_diff(
                                output[batch_id, seq_id, imu_id], centroid))
                        target_distances.append(
                            self.get_angle_diff(
                                target[batch_id, seq_id, imu_id], centroid))
                    output_label = np.argmin(np.array(output_distances))
                    output_labels[batch_id, seq_id, imu_id, output_label] = 1
                    target_labels[batch_id, seq_id, imu_id] = int(
                        np.argmin(np.array(target_distances)))
        self.metric.record_output(output_labels, output_indices, target_labels,
                                  prev_absolute_imu, next_absolute_imu,
                                  batch_size)

    def get_angle_diff(self, q1, q2):
        q1 = Quaternion(q1)
        q2 = Quaternion(q2)
        return math.acos(2 * (np.dot(q1.normalised.q, q2.normalised.q)**2) - 1)

    def report(self):
        return self.metric.report()

    def final_report(self):
        return self.metric.final_report()


class AngleEvaluationMetric(BaseMetric):

    def __init__(self, args):
        self.centroid = args.relative_centroids
        self.regression = args.regression
        self.absolute_regress = args.absolute_regress
        self.meter = AverageMeter()
        self.args = args
        self.stats = []
        self.outputs = []
        self.targets = []
        self.visualization = (args.mode == 'test')

    def record_output(self, output, output_indices, target, prev_absolute_imu,
                      next_absolute_imu, batch_size=1):

        if self.args.no_angle_metric:
            return []

        prev_absolute_imu = prev_absolute_imu[:, output_indices]
        next_absolute_imu = next_absolute_imu[:, output_indices]

        assert output.dim() == 4
        if self.regression:
            # Output is the vectors itself.
            relative_imus = output
        else:
            output = torch.max(output, -1)[1]  #get labels from vectors
            relative_imus = self.get_diff_from_initial(output)
        if self.absolute_regress:
            resulting_imus = output
        else:
            resulting_imus = self.inverse_subtract(prev_absolute_imu,
                                                   relative_imus)

        angle_diff = self.get_angle_diff(next_absolute_imu, resulting_imus)
        whole_diff = self.get_angle_diff(next_absolute_imu[:, 0:1, :, :],
                                         next_absolute_imu[:, -1:, :, :])

        self.stats.append([whole_diff, angle_diff.mean()])
        self.outputs.append(output)
        self.targets.append(target)

        degree = angle_diff * 180 / math.pi
        self.meter.update(degree, batch_size)

        return resulting_imus

    def inverse_subtract(self, initial_absolute, relative):
        size = relative.size()
        result = np.zeros(size)
        #the following for loops should be optimized
        for batch_id in range(size[0]):
            for seq_id in range(size[1]):
                for imu_id in range(size[2]):

                    if seq_id == 0:
                        q1 = Quaternion(
                            initial_absolute[batch_id, seq_id, imu_id])
                    else:
                        q1 = Quaternion(result[batch_id, seq_id - 1, imu_id])

                    diff = Quaternion(relative[batch_id, seq_id, imu_id])
                    q2 = (q1.normalised * diff.normalised).normalised.q
                    result[batch_id, seq_id, imu_id] = q2

        return result

    def get_angle_diff(self, target, result):
        size = target.size()
        sequence_length = size[1]
        all_averages = np.zeros((sequence_length)).astype(np.float)
        for seq_id in range(sequence_length):
            average = AverageMeter()
            for batch_id in range(size[0]):
                for imu_id in range(size[2]):
                    goal = Quaternion(target[batch_id, seq_id, imu_id])
                    out = Quaternion(result[batch_id, seq_id, imu_id])
                    acos = (2 * (np.dot(out.normalised.q, goal.normalised.q)**2)
                            - 1)
                    acos = round(acos, 6)
                    if acos > 1 or acos < -1:
                        pdb.set_trace()
                    radian = math.acos(acos)
                    average.update(radian)

            all_averages[seq_id] = (average.avg)

        return all_averages

    def report(self):
        result = 'Angle_Metric '
        for i in range(len(self.meter.val)):
            avg = self.meter.avg[i]
            val = self.meter.val[i]
            result += ('seq_id {id}: {val:.2f} ({avg:.2f}) | '.format(
                avg=avg, val=val, id=i))
        return result + 'And the mean ' + str(self.meter.avg.mean())

    def final_report(self):
        return self.report()

    def get_diff_from_initial(self, class_label):
        size = class_label.size()
        relative_imus = []
        for imu_id in range(size[2]):
            centroid_indices = class_label[:, :, imu_id].long().cpu().view(-1)
            relative = self.centroid[imu_id][centroid_indices].view(
                size[0], size[1], -1)
            relative_imus.append(relative)
        relative_imus = torch.stack(relative_imus).transpose(0, 1).transpose(
            1, 2)
        return relative_imus


class IouSegmentation(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()
        self.stat = AverageMeter()

    def record_output(self, output, output_indices, target, prev_absolute_imu,
                      next_absolute_imu, batch_size=1):
        cleaned = self.clean_mask(output)
        iou = self.calc_iou(cleaned, target)
        self.meter.update(iou * 100, batch_size)

    def report(self):
        return ('IOU_Metric {meter.val:.2f} ({meter.avg:.2f})'.format(
            meter=self.meter))

    def final_report(self):
        return ('IOU_Metric  {meter.avg:.2f}'.format(meter=self.meter))

    def clean_mask(self, output):
        cleaned = output.clone()
        cleaned[output > 0.5] = 1.0
        cleaned[output < 0.5] = 0.0
        return cleaned

    def calc_stat(self, target):
        ones = torch.sum((target == 1).view(target.size(0), -1).float(), 1)
        zeros = torch.sum((target == 0).view(target.size(0), -1).float(), 1)
        return (ones / (ones + zeros)).mean()

    def calc_iou(self, cleaned, target):
        eps = 1e-6
        intersection = ((cleaned + target) >= 2).float()
        union = ((cleaned + target) >= 1).float()
        sum_intersection = torch.sum(
            intersection.view(intersection.size(0), -1), 1)
        sum_union = torch.sum(union.view(union.size(0), -1), 1)
        return (sum_intersection / (sum_union + eps)).mean()


class ClassificationMetric(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()

    def record_output(self, output, output_indices, target, prev_absolute_imu,
                      next_absolute_imu, batch_size=1):
        prediction = torch.max(output, 1)[1]
        success = torch.sum((prediction == target).float()) / len(prediction)
        self.meter.update(success * 100, batch_size)

    def report(self):
        return ('Mean Accuracy {meter.val:.2f} ({meter.avg:.2f})'.format(
            meter=self.meter))

    def final_report(self):
        return ('Mean Accuracy  {meter.avg:.2f}'.format(meter=self.meter))


class AllAtOnce(BaseMetric):

    def __init__(self, args):
        self.args = args
        self.meter = AverageMeter()
        self.confusion = None

    def record_output(self, output, output_indices, target, prev_absolutes,
                      next_absolutes, batch_size=1):
        assert output.dim() == 4
        assert target.dim() == 3

        _, predictions = output.max(3)

        # Compute per class accuracy for unbalanced data.
        sequence_length = output.size(1)
        num_label = output.size(2)
        num_class = output.size(3)
        correct_alljoint = (target == predictions).float().sum(2)
        sum_of_corrects = correct_alljoint.sum(1)
        max_value = num_label * sequence_length
        count_correct = (sum_of_corrects == max_value).float().mean()
        correct_per_seq = ((correct_alljoint == num_label - 1).sum(1).float() /
                           sequence_length).mean()
        self.meter.update(
            torch.Tensor([count_correct * 100, correct_per_seq * 100]),
            batch_size)

    def report(self):
        return ('AllAtOnce {meter.val[0]:.2f} ({meter.avg[0]:.2f})     ' +
                'PerSeq {meter.val[1]:.2f} ({meter.avg[1]:.2f})'
               ).format(meter=self.meter)

    def final_report(self):
        return self.report()
