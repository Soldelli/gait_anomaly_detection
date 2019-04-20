function [y_space] = space(input, Fs, cutOff)

Wn = double(ldivide(ldivide(2,Fs),cutOff));
[B, A] = butter(4, Wn, 'high');
x = (1:length(input)) ./ Fs; 
y_acc = detrend(input);
y_acc = filtfilt(B, A, y_acc);
y_vel = cumtrapz(x, y_acc);
y_vel = filtfilt(B, A, y_vel);
y_space = cumtrapz(x, y_vel);


