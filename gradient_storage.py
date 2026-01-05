class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module.
    """
    def __init__(self, module, num_adv_passage_tokens):
        self._stored_gradient = None
        self.num_adv_passage_tokens = num_adv_passage_tokens
        module.register_full_backward_hook(self.hook)
        # self.call_count = 0

    def hook(self, module, grad_in, grad_out):
        # self.call_count += 1
        grad = grad_out[0][:, -self.num_adv_passage_tokens:]
        
        # if grad.dim() == 3:
            # containing batch
            # grad_sum = grad
        grad_sum = grad.sum(dim=0)
        if self._stored_gradient is None:
            self._stored_gradient = grad_sum
        else:
            self._stored_gradient += grad_sum

    def get(self):
        return self._stored_gradient

    # 【新增这个方法】用来手动清空梯度
    def zero_grad(self):
        # return
        # self.call_count = 0
        self._stored_gradient = None

    # def get_call_count(self):
        # following the original paper, this should increase (BUG)
        # return self.call_count