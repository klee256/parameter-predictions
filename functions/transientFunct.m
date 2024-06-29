clc
clear all
close all

%% tc

%pc=parcluster('local');
%parpool(pc,48)

%load P58_6tc2024.mat
%load N57_7tc2024.mat

% Multiple

%data =  N57_7tc;
% P58_6deltaQ = zeros(1,size(data,2));
% timeDiv = 1e-6;
% 
% for i = 1:size(data,2)
%     P58_6deltaQ(i) = tcFit(timeDiv,data(:,i));
%     disp(i)
% end
% 
% %save('P58_6deltaQ2024.mat','P58_6deltaQ')
% 
% %visualizeMap(N57_7deltaQ,'\Delta Q (C)',70,0.025)
% %saveas(gcf,'P58_6deltaQ map.png');

% single sample testing
%sample = 1568;
% %figure(); 
% %plot(P58_6tc(:,sample))
% 
%timeDiv = 1e-6;
%t = linspace(0,10*timeDiv,2500).';
% %t = linspace(0,10,2500).';
 
%s = N57_7tc(:,sample);
%s = P58_6tc(:,sample);
%s = s./490;
%s = s - mean(s(2000:2500));
%s = s - mean(s);

% [~,peakIndex] = max(smooth(s,50));
% 
% halfT = t(peakIndex:end);
% halfS = s(peakIndex:end);
% f = fit(halfT, halfS, 'exp2');
% fitted_curve = f(halfT);

%f = fit(t,s,'exp2');
%fitted_curve = f(t);
% 
% h=figure();
% 
% set(h, 'Position', [100, 100, 800, 400]);
% 
% plot(t,s)
% hold on
% %plot(halfT,fitted_curve)
% plot(halfT,fitted_curve, 'LineWidth', 2);
% hold on
% yline(0, 'LineWidth', 2);
% xlabel('time (s)')
% ylabel('current (A)')
% legend('raw','fit')
% 


% validRange = fitted_curve > 0;
% 
% highResTime = linspace(min(halfT(validRange)),max(halfT(validRange)),10000);
% interpVal = interp1(halfT(validRange),fitted_curve(validRange),highResTime);
% 
% % figure();
% % plot(halfT,fitted_curve)
% % hold on
% % plot(highResTime,interpVal,'r')
% % hold on
% % plot(t,s)
% 
% charge = trapz(halfT(validRange),fitted_curve(validRange));

%poolobj = gcp('nocreate');
%delete(poolobj);

%% tv

%load N57_7tv.mat
%data =  N57_7tv;

data = readmatrix('N17_6 tv.xlsx');

deltaV = zeros(1,size(data,2));
timeDiv = 1e-3;
%timeDiv = 1e-3;

for i = 1:size(data,2)
     deltaV(i) = tvFit(timeDiv,data(:,i));
end

%save('N57_7deltaV2024.mat','N57_7deltaV')
visualizeMap(N17_6deltaV,'deltaV',50,0.025)
visualizeGrid(deltaV,'delV',80)

function deltaQ = tcFit(timeDiv,tcData)

    % tcData should be a single 2500x1 
    
    t = linspace(0,10*timeDiv,2500).';
    
    s = tcData./490;
    s = s - mean(s(2000:2500));
    
    forPeak = smooth(s,50);
    forPeak(1:250) = zeros(1,250);
    forPeak(2001:2500) = zeros(1,500);

    [~,peakIndex] = max(forPeak);
    
    %figure(); plot(smooth(s,50));

    halfT = t(peakIndex:end);
    halfS = s(peakIndex:end);
    
    % method 1
    f = fit(halfT, halfS, 'exp2');
    fitted_curve = f(halfT);
    
    % method 2
    % double_exp = @(params, x) params(1) * exp(params(2) * x) + params(3) * exp(params(4) * x);
    % initial_guesses = [0, -4e5, 8e-5, -4e5]; % for N57_7
    % initial_guesses = [10e-6, -8e5, -2e-7, -3e5]; % for P58_6
    % options = optimoptions('lsqcurvefit', 'Display', 'none');
    % params_fit = lsqcurvefit(double_exp, initial_guesses, halfT, halfS, [], [], options);
    % fitted_curve = double_exp(params_fit,halfT);

    %figure();
    %plot(halfT,fitted_curve)
    %hold on
    %plot(t,s)

    validRange = fitted_curve > 0;

    highResTime = linspace(min(halfT(validRange)),max(halfT(validRange)),10000);
    interpVal = interp1(halfT(validRange),fitted_curve(validRange),highResTime);

    deltaQ = trapz(highResTime,interpVal);

end


function deltaV = tvFit(timeDiv,tvData)

    % tvData should be a single 2500x1 
    
    t = linspace(0,10*timeDiv,2500).';
    
    s = tvData - mean(tvData(2000:2500));
    
    forPeak = smooth(s,50);
    forPeak(1:250) = zeros(1,250);
    forPeak(2001:2500) = zeros(1,500);

    [~,peakIndex] = max(forPeak);
    
    %figure(); plot(smooth(s,50));

    halfT = t(peakIndex:end);
    halfS = s(peakIndex:end);
    
    % method 1
    f = fit(halfT, halfS, 'exp2');
    fitted_curve = f(halfT);
    
    % method 2
    % double_exp = @(params, x) params(1) * exp(params(2) * x) + params(3) * exp(params(4) * x);
    % initial_guesses = [0, -4e5, 8e-5, -4e5]; % for N57_7
    % %initial_guesses = [10e-6, -8e5, -2e-7, -3e5]; % for P58_6
    % options = optimoptions('lsqcurvefit', 'Display', 'none');
    % params_fit = lsqcurvefit(double_exp, initial_guesses, halfT, halfS, [], [], options);
    % fitted_curve = double_exp(params_fit,halfT);

    % figure();
    % plot(halfT,fitted_curve,'LineWidth',1.5)
    % hold on
    % plot(t,s)
    % hold on
    % yline(0)
    % xlabel('time (s)')
    % ylabel('voltage (V)')
    % legend('fit','raw')
    % axis tight
    % grid on

    deltaV = max(fitted_curve);

    if deltaV < 0
        deltaV = max(forPeak);
        if deltaV < 0
            deltaV = 0;
            disp('big error')
        end
        disp('hello')
    end

end
