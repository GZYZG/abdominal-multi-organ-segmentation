from medpy import metric as Metrics

class Evaluator:
    def __init__(self, metrics: list):
        self.metrics = metrics

    def eval(self, y_true, y_pred, metrics=None):
        """
        给定真实标签和预测值，进行评估
        :param y_true: 真实标签。形状 (batch, ...)
        :param y_pred: 预测值。形状 (batch, ...)
        :param metrics: 可以指定评测指标，否则使用初始化时的指标

        Return: dict-like。键为评测指标，值为对应的评测值，为列表，每个元素对应一个样本
        """
        selected_metrics = metrics if metrics is not None else self.metrics

        results = {k: None for k in selected_metrics}

        for metric in selected_metrics:
            if metric == 'acc':
                results[metric] = self._eval_acc(y_true, y_pred)
            elif metric == 'precision':
                pass

        return results

    def _eval_acc(self, y_true, y_pred):
        """
        给定真实标签和预测值，进行评估 accuracy
        :param y_true: 真实标签。形状 (batch, ...)
        :param y_pred: 预测值。形状 (batch, ...)
        :param metrics: 可以指定评测指标，否则使用初始化时的指标

        Return: dict-like。键为评测指标，值为对应的评测值，为列表，每个元素对应一个样本
        """
        t = (y_true == y_pred).sum()

        pass

    def _eval_precision(self, y_true, y_pred):
        """
        给定真实标签和预测值，进行评估 precision
        :param y_true: 真实标签。形状 (batch, ...)
        :param y_pred: 预测值。形状 (batch, ...)
        :param metrics: 可以指定评测指标，否则使用初始化时的指标

        Return: dict-like。键为评测指标，值为对应的评测值，为列表，每个元素对应一个样本
        """
        pass

    def _eval_IoU(self, y_true, y_pred):
        """
        给定真实标签和预测值，进行评估 mIoU
        :param y_true: 真实标签。形状 (batch, ...)
        :param y_pred: 预测值。形状 (batch, ...)
        :param metrics: 可以指定评测指标，否则使用初始化时的指标

        Return: dict-like。键为评测指标，值为对应的评测值，为列表，每个元素对应一个样本
        """
        pass

    def _eval_dice(self, y_true, y_pred):
        """
        给定真实标签和预测值，进行评估 dice coefficient
        :param y_true: 真实标签。形状 (batch, ...)
        :param y_pred: 预测值。形状 (batch, ...)
        :param metrics: 可以指定评测指标，否则使用初始化时的指标

        Return: dict-like。键为评测指标，值为对应的评测值，为列表，每个元素对应一个样本
        """
        pass


if __name__ == "__main__":
    evalor = Evaluator(metrics=['precision', 'acc'])
    y_pred = []
    y_true = []
    ret = evalor.eval(y_true, y_pred)

    for k, v in ret.items():
        print(f"{k} : {v}")
