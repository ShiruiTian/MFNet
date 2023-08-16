x = [0:0.1:10];
y = [0:0.1:10];

[X,Y] = meshgrid(x,y);
Z = exp(-(X-5).^2/10 - (Y-5).^2/10);
% Define the Gaussian function
fun = @(params,X) params(1) * exp(-(X(:,1)-params(2)).^2/params(3) - (X(:,2)-params(4)).^2/params(5));
% Initial quess for parameters
paramse = [1 5 5 5 107];
% Perform the curve fit
params = lsqcurvefit(fun, paramse,[X(:) Y(:)],Z(:));