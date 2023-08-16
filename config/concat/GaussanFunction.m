load('gaussian_parameters1.mat')

X = -20 : 0.01 : 20;
Y = -20 : 0.01 : 20;
 
 
[ XX, YY ] = meshgrid( X, Y );
Z = (( XX - x0 ).^2) / (sigmax^2) + (( YY - y0 ).^2) / (sigmay^2);
Z = exp(-Z);


figure, mesh(X, Y, Z); % Ïß¿òÍ¼
