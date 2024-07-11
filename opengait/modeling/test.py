class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model  # 用于储存模型
        self.target_layer = target_layer  # 目标层的名称
        self.gradients = None  # 最终的梯度图

    def resume_ckpt(self, restore_hint):
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
        if isinstance(restore_hint, int):
            save_name = "baseline_woBnneck"
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def save_gradient(self, grad):
        self.gradients = grad  # 用于保存目标特征图的梯度（因为pytorch只保存输出，相对于输入层的梯度
        # ，中间隐藏层的梯度将会被丢弃，用来节省内存。如果想要保存中间梯度，必须
        # 使用register_hook配合对应的保存函数使用，这里这个函数就是对应的保存
        # 函数其含义是将梯度图保存到变量self.gradients中，关于register_hook
        # 的使用方法我会在开一个专门的专题，这里不再详述

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """

        conv_output = None
        for name, layer in self.model._modules.items():
            if name == "fc":
                break
            x = layer(x)
            if name == self.target_layer:
                conv_output = x  # 将目标特征图保存到conv_output中
                x.register_hook(self.save_gradient)  # 设置将目标特征图的梯度保存到self.gradients中
        return conv_output, x  # x为最后一层特征图的结果

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)  # 用于提取特征图与梯度图

    def generate_cam(self, input_image, target_class=None):
        # 1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # one hot编码，令目标类置1
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
        target = conv_output.data.numpy()[0]
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.model.zero_grad()
        # 步骤1.2.2 计算反向传播
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # 步骤1.2.3 获取目标层各特征图的梯度
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # 初始化热力图
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # 步骤2.2 计算各特征图的加权值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        # 步骤2.3 对热力图进行后处理，即将结果变换到0~255
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS)) / 255
        return cam