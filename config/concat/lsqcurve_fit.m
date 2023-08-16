% 读取保存的高斯参数
load('gaussian_parameters.mat');

% Load the x and y coordinates from the Python-generated file
data = load('data/pc_rgb/foreground_points_FPS32.txt');
xdata = data(:, 1);
ydata = data(:, 2);

% 定义拟合函数
fun = @(x, xdata) exp(-((xdata - x(3)).^2) / (x(1)^2)) + exp(-((xdata - x(4)).^2) / (x(2)^2));

% 设置lsqcurvefit函数的选项
options = optimoptions('lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'MaxFunEvals', 1000);

% 调用lsqcurvefit进行拟合
x = lsqcurvefit(fun, [sigmax, sigmay, x0, y0], xdata, ydata, [], [], options);

% 生成拟合函数的曲线
xx = linspace(0, 2*pi, 150);
yy = fun(x, xx);

% 绘制原始数据散点图和拟合函数曲线
plot(xdata, ydata, 'o');
hold on
plot(xx, yy);

% 绘制拟合函数的等高线
[X, Y] = meshgrid(xx, yy);
Z = exp(-((X - x(3)).^2) / (x(1)^2)) + exp(-((X - x(4)).^2) / (x(2)^2));
contour(X, Y, Z);

% 添加图例
%legend('原始数据', '拟合函数', '拟合函数等高线');

% 设置横纵坐标标签
xlabel('xdata');
ylabel('ydata');
