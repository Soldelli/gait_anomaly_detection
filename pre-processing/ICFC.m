% Function that returns Initial Contacts (IC) and Final Contacts (Fc) of
% each step
% input: input signal - should be accelereation y component
% Fs : sampling frequency
% scale : cwt scale

function [locsIC, locsFC] = ICFC(input, Fs, scale, visualize)

% detrend
in_detrend = detrend(input);
%in_detrend = in_detrend + 0.2*randn(1,length(in_detrend));
% LPF - 20Hz - 4th order
Wn = double(ldivide(ldivide(2,Fs),20));
[B, A] = butter(4, Wn, 'low');
in_filt = filtfilt(B, A, in_detrend);

%figure;plot(in_detrend,'k'); hold on; plot(in_filt,'k-.');
%legend('Original signal', 'Filtered signal');
%set(findall(gcf,'-property','FontSize'),'FontSize', 25);

% integration
x = (1:length(in_filt)) ./ Fs;
in_integr = cumtrapz(x, in_filt);

% 1st gaussian cwt
gcwt1 = cwt(in_integr, scale, 'gaus1');

[~, locsIC] = findpeaks(-gcwt1);
%figure; plot(locsIC, gcwt1(locsIC),'ko'); hold on; plot(gcwt1)

% 2nd gaussian cwt
gcwt2 = cwt(gcwt1, scale, 'gaus1');

[~, locsFC] = findpeaks(gcwt2);
%figure; plot(locsFC,gcwt2(locsFC),'ko'); hold on; plot(gcwt2)

% IC correction
minDist = 0.25 * Fs; % 0.25
maxDist = 2.25 * Fs; % 2.25

diff1 = locsIC(2:end) - locsIC(1:end-1);
temp = [];
for i = 1 : length(diff1)
    if diff1(i) < minDist || diff1(i) > maxDist
        temp = [temp i+1];
    end
end
locsIC(temp) = 0;
locsIC = locsIC(locsIC~=0);

%diff2 = locsFC(2:end) - locsFC(1:end-1);
%temp = [];
%for i = 1 : length(diff2)
%    if diff2(i) < minDist | diff1(i) > maxDist
%        temp = [temp i+1];
%    end
%end
%locsFC(temp) = 0;
%locsFC = locsFC(locsFC~=0);

%n = min(length(locsFC), length(locsIC));
%locsIC = locsIC(1:n);
%locsFC = locsFC(1:n);

if visualize
    % plot
    figure; plot(in_detrend,'k'); hold on; plot(gcwt1,'k--'); hold on; plot(gcwt2,'k-.'); ...
        hold on; plot(locsIC, gcwt1(locsIC),'ko','MarkerSize',15); hold on; plot(locsFC, gcwt2(locsFC),'k^','MarkerSize',15);
    legend('detrended input','1st gcwt', '2nd gcwt', 'IC', 'FC', 'Location', 'SouthEast');
    ylabel('m/s^2'); xlabel('samples');
    
    set(findall(gcf,'-property','FontSize'),'FontSize', 25);
end

