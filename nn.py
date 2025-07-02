import torch
from torch.nn import Module, ModuleList, Conv1d, MaxPool2d, Upsample, BatchNorm1d, SiLU, Linear, Sigmoid, Dropout

import time

class ArtificialNuclear:
      """
      Сознание CARUS v0.1
      """
      def __init__(self, internal_neuron, input_neuron, output_neuron, device="cpu", lr=10e-1, max_delay=0.05, activation_level=50, activation_level_input=20, activation_level_output=70, max_recharge_rate=0.2):
            self.internal_neuron = internal_neuron
            self.input_neuron = input_neuron
            self.output_neuron = output_neuron

            self.lr = lr
            self.max_delay = max_delay
            self.activation_level = activation_level
            self.activation_level_input = activation_level_input
            self.activation_level_output = activation_level_output
            self.max_recharge_rate = max_recharge_rate
            self.device = device

            self.f_first_iteration = True

            self.__init__tensor()

      def __init__tensor(self):
            self.engram = torch.rand(size=(self.internal_neuron + self.input_neuron, self.internal_neuron + self.output_neuron), dtype=torch.float32).to(self.device) * 2 - 1
            self.horizontal_connections = {
                "input": torch.rand(size=(self.input_neuron, self.input_neuron), dtype=torch.float32).to(self.device),
                "output": torch.rand(size=(self.output_neuron, self.output_neuron), dtype=torch.float32).to(self.device)
            }

            self.internal_state = torch.rand(size=(1, self.internal_neuron), dtype=torch.float32).to(self.device)

            self.activation_level = {
                "internal": torch.rand(size=(1, self.internal_neuron + self.output_neuron), dtype=torch.float32).to(self.device) * self.activation_level,
                "input": torch.rand(size=(1, self.input_neuron), dtype=torch.float32).to(self.device) * self.activation_level_input,
                "output": torch.rand(size=(1, self.output_neuron), dtype=torch.float32).to(self.device) * self.activation_level_output
            }

            self.delay = torch.rand(size=(1, self.internal_neuron + self.output_neuron), dtype=torch.float32).to(self.device) * self.max_delay  # Задержка между спайками
            self.time_between_spike = torch.zeros(size=(1, self.internal_neuron + self.output_neuron), dtype=torch.float32).to(self.device)  # Текущие время прошедшие после прошлого спайка
            self.recharge_rate = torch.rand(size=(1, self.internal_neuron + self.output_neuron), dtype=torch.float32).to(self.device) * self.max_recharge_rate  # Скорость перезарядки спайка

      def __call__(self, input:torch.Tensor):
            """
            Один шаг итерации
            """
            if input.dim() != 2:
              raise ValueError(f"Tensor must have 2 dim, but got {input.dim()}")
            if self.f_first_iteration:
              self.start_time_iteration = time.perf_counter()
              self.f_first_iteration = False

            # Горизонтальные связи
            input = self.__pre_and_postphase(input, mode="input").float()

            # Фаза внутреннего обдумывания
            output = self.__internal_phase(input)

            # Горизонтальные связи
            output = self.__pre_and_postphase(output, mode="output")

            # Отключение связей
            # Добавление случайных связей

            return output

      def __pre_and_postphase(self, input, mode):
            input = torch.matmul(input.float(), self.horizontal_connections[mode])
            input = input >= self.activation_level[mode]  # Проверка активации нейрона
            # delay для входа/выхода
            return input

      def __internal_phase(self, input):
            # Расчет следующего состояния внутренних нейронов
            self.internal_state = torch.concatenate(tensors=[input, self.internal_state], dim=1)  # Соединение входных и внутренних нейронов
            self.tmp = self.internal_state.clone().float()  # Копируем для будущего обучения
            self.internal_state = torch.matmul(self.internal_state, self.engram)  # Расчет передачи сигнала через энграммы
            activation_neuron = self.internal_state >= 0  # Нейроны генерируют на аксоне только положительную деполяризацию
            self.internal_state = torch.mul(self.internal_state, activation_neuron)
            self.internal_state = self.internal_state >= self.activation_level["internal"]  # Проверка активации нейрона

            # Расчет, что спайк перезарядился и обновление состояния перезарядки
            self.time_between_spike = self.time_between_spike + self.recharge_rate * (time.perf_counter() - self.start_time_iteration)  # Добавление времени к перезарядке
            self.start_time_iteration = time.perf_counter()  # Обнуление счетчика таймера иттерации
            activation_neuron = self.time_between_spike >= self.delay  # Проверка тех нейронов, что готовы к спайку

            # Активация спайка
            self.internal_state = torch.mul(self.internal_state, activation_neuron).float()  # Окончательный спайк

            # STDP
            self.__STDP()

            # Перезарядка спайка
            self.time_between_spike = torch.mul(self.internal_state == False, self.time_between_spike)

            # Отделяем выходные нейроны
            self.internal_state, output = torch.split(tensor=self.internal_state, split_size_or_sections=[self.internal_neuron, self.output_neuron], dim=1)
            self.internal_state = self.internal_state.float()
            return output

      def __STDP(self):
            activation_synapses_neurons = torch.matmul(torch.transpose(self.tmp, dim0=1, dim1=0), self.internal_state)  # Определение задействовавшихся дендридов
            self.engram = self.engram + activation_synapses_neurons * self.lr
            mean = self.engram.mean()
            std = self.engram.std()
            self.engram = (self.engram - mean) / std


class Conv(Module):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel: int = 3,
               stride: int = 1,
               padding: int = 0,
               activation: bool = True):
    super(Conv, self).__init__()
    self.act = activation

    self.convolutional = Conv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel,
      stride=stride,
      padding=padding,
      bias=False
    )
    self.batch = BatchNorm1d(num_features=out_channels)
    self.fn_act = SiLU()
    self.dropout = Dropout(p=0.2)

  def forward(self, x):
    x = self.convolutional(x)
    x = self.batch(x)
    if self.act:
      x = self.fn_act(x)
    x = self.dropout(x)
    return x


class Bottleneck(Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 3):
      super(Bottleneck, self).__init__()

      bottle_neck = in_channels // 2

      self.layer_in = Conv(
        in_channels=in_channels,
        out_channels=bottle_neck,
        kernel=1,
        stride=1,
        padding=0
      )
      self.layer_out = Conv(
        in_channels=bottle_neck,
        out_channels=in_channels,
        kernel=1,
        stride=1,
        padding=0
      )

    def forward(self, x):
      residual = x.clone()

      x = self.layer_in(x)
      x = self.layer_out(x)

      x = x + residual

      del residual
      return x


class C2F(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_bottle_neck: int = 3,
            shortcut: bool = True
    ):
        super(C2F, self).__init__()

        self.shortcut = shortcut

        self.conv = Conv(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel=1,
          stride=1,
          padding=0
        )
        self.layers = []
        self.split_list = []
        self.channels_list = []
        for _ in range(num_bottle_neck):
          self.split_list.append(out_channels // 2)
          out_channels = out_channels - out_channels // 2
          self.channels_list.append(out_channels)
          self.layers.append(Bottleneck(in_channels=out_channels))
        self.layers = ModuleList(self.layers)

    def forward(self, x: torch.Tensor):
        residual = []
        res = x.clone()
        x = self.conv(x)

        for layer, split_channels, channels in zip(self.layers, self.split_list, self.channels_list):
          x = torch.split(x, split_size_or_sections=[split_channels, channels], dim=1)
          residual.append(x[0])
          x = x[1]
          x = layer(x)

        residual.append(x)
        x = torch.concatenate(residual, dim=1)
        if self.shortcut:
          x = x + res
        return x


class SPPF(Module):
    def __init__(
            self,
            channels: int,
            kernel: int,
            padding: int,
            num_maxpool: int = 3
    ):
        super(SPPF, self).__init__()

        self.in_conv = Conv(
          in_channels=channels,
          out_channels=channels,
          kernel=kernel,
          stride=1,
          padding=padding
        )
        self.out_conv = Conv(
          in_channels=channels,
          out_channels=channels,
          kernel=kernel,
          stride=1,
          padding=padding
        )

        self.layers = []
        self.channels_list = []
        self.split_list = []
        for i in range(num_maxpool):
          self.split_list.append(channels // 2)
          channels = channels - channels // 2
          self.channels_list.append(channels)
          self.layers.append(MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.layers = ModuleList(self.layers)

    def forward(self, x):
        x = self.in_conv(x)
        residual = []
        for layer, split, channels in zip(self.layers, self.split_list, self.channels_list):
          x = torch.split(x, (split, channels), dim=1)
          residual.append(x[0])
          x = x[1]
          x = layer(x)
        residual.append(x)
        x = torch.concatenate(residual, dim=1)
        x = self.out_conv(x)
        return x


class Backbone(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: list,
            extractor_depth: int
    ):
        super(Backbone, self).__init__()

        self.in_conv = Conv(
          in_channels=in_channels,
          out_channels=out_channels[0],
          kernel=3,
          stride=2,
          padding=1
        )

        self.layer = []
        for i in range(0, len(out_channels) - 1):
          self.layer.append(
            Conv(
              in_channels=out_channels[i],
              out_channels=out_channels[i + 1],
              kernel=3,
              stride=2,
              padding=1
            )
          )
          self.layer.append(
            C2F(
              in_channels=out_channels[i + 1],
              out_channels=out_channels[i + 1],
              num_bottle_neck=3 * i // 2,
              shortcut=True
            )
          )
        self.layer = ModuleList(self.layer)

    def forward(self, x):
        out = []
        x = self.in_conv(x)
        for layer in self.layer:
            x = layer(x)
            if type(layer) == C2F:
                out.append(x)
        return out


class Neck(Module):
    def __init__(
            self,
            channels: list,
            kernel: int = 5,
            padding: int = 2,
            scale_factor: int = 2
    ):
        super(Neck, self).__init__()
        self.sppf = SPPF(channels=channels[-1], kernel=kernel, padding=padding)
        self.upsample_down = Upsample(scale_factor=scale_factor)

        self.c2f_down = C2F(in_channels=channels[-1] + channels[-2], out_channels=channels[-2], num_bottle_neck=3,
                            shortcut=False)
        self.upsample_up = Upsample(scale_factor=scale_factor)

        self.c2f_up = C2F(in_channels=channels[-2] + channels[-3], out_channels=channels[-3], num_bottle_neck=3,
                          shortcut=False)
        self.conv_up = Conv(in_channels=channels[-3], out_channels=channels[-3], stride=2, padding=1)
        self.c2f_up_back = C2F(in_channels=channels[-2] + channels[-3], out_channels=channels[-2], num_bottle_neck=3,
                               shortcut=False)

        self.conv_down = Conv(in_channels=channels[-2], out_channels=channels[-2], stride=2, padding=1)
        self.c2f_down_back = C2F(in_channels=channels[-1] + channels[-2], out_channels=channels[-1], num_bottle_neck=3,
                                 shortcut=False)

        self. fnn = Linear(in_features=2560, out_features=1)
        self.final_activation = Sigmoid()

    def forward(self, x):
        high_resolution, med_resolution, low_resolution = x[1], x[2], x[3]

        x = self.sppf(low_resolution)
        x_2 = x.clone()
        x = self.upsample_down(x)
        x = torch.concatenate([x, med_resolution], dim=1)

        x = self.c2f_down(x)
        x_1 = x.clone()
        x = self.upsample_up(x)
        x = torch.concatenate([x, high_resolution], dim=1)

        to_detect = self.c2f_up(x)

        x = to_detect.clone()
        x = self.conv_up(x)
        x = torch.concatenate([x, x_1], dim=1)
        x = self.c2f_up_back(x)

        x = self.conv_down(x)
        x = torch.concatenate([x, x_2], dim=1)
        x = self.c2f_down_back(x)

        x = x.reshape(x.size(0), 1, -1)
        x = self.fnn(x)
        x = self.final_activation(x)
        return x