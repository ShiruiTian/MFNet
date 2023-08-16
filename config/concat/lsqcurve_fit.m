% ��ȡ����ĸ�˹����
load('gaussian_parameters.mat');

% Load the x and y coordinates from the Python-generated file
data = load('data/pc_rgb/foreground_points_FPS32.txt');
xdata = data(:, 1);
ydata = data(:, 2);

% ������Ϻ���
fun = @(x, xdata) exp(-((xdata - x(3)).^2) / (x(1)^2)) + exp(-((xdata - x(4)).^2) / (x(2)^2));

% ����lsqcurvefit������ѡ��
options = optimoptions('lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'MaxFunEvals', 1000);

% ����lsqcurvefit�������
x = lsqcurvefit(fun, [sigmax, sigmay, x0, y0], xdata, ydata, [], [], options);

% ������Ϻ���������
xx = linspace(0, 2*pi, 150);
yy = fun(x, xx);

% ����ԭʼ����ɢ��ͼ����Ϻ�������
plot(xdata, ydata, 'o');
hold on
plot(xx, yy);

% ������Ϻ����ĵȸ���
[X, Y] = meshgrid(xx, yy);
Z = exp(-((X - x(3)).^2) / (x(1)^2)) + exp(-((X - x(4)).^2) / (x(2)^2));
contour(X, Y, Z);

% ���ͼ��
%legend('ԭʼ����', '��Ϻ���', '��Ϻ����ȸ���');

% ���ú��������ǩ
xlabel('xdata');
ylabel('ydata');
