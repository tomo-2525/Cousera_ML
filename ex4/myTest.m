data = imread("/mnt/c/Users/CRAFIS/Downloads/Data.bmp");
data = double(data);
tmp = data(:)';
displayData(tmp(1,:));
print("/mnt/c/Users/CRAFIS/Downloads/Data.png");
pred = predict(Theta1, Theta2, tmp(1,:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
clear tmp;
