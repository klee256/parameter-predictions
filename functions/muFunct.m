% muRedo.m

clc
clear all
close all

addpath(genpath(fileparts(pwd)));

load('P61_11m0.mat')
load('P61_11m1.mat');
load('P61_11m2.mat');
load('P61_11m3.mat');
times=[2.5e-6,2.5e-6,1.0e-6,2.5e-6];
biases=[0,0.1,0.2,0.3];

%sampleNum = 141;
%mySample = P61_11m0(:,sampleNum);

y_axis0 = mobilityFit(times(1),P61_11m0);
%y_axis1 = mobilityFit(times(2),P61_11m1);
%y_axis2 = mobilityFit(times(3),P61_11m2);
%y_axis3 = mobilityFit(times(4),P61_11m3);

load 'y_axis0.mat'
load 'y_axis1.mat'
load 'y_axis2.mat'
load 'y_axis3.mat'

mu = zeros(1,size(y_axis0,2));
for i = 1:length(mu)
    temp=[y_axis0(i); y_axis1(i); y_axis2(i); y_axis3(i)];
    slope=polyfit(biases,temp,1);
    mu(i)=slope(1);
end

function y_axis = mobilityFit(tS,data)

    d=400e-9; d2=(d*100)^2; % film thickness and its square (cm^2)
    one_e=1/(exp(1).^2);

    time = linspace(0,10*tS,2500).';
    highResTime = linspace(0,10*tS,25000);

    y_axis = zeros(1,size(data,2));

    %figure();
    for i = 1:size(data,2)
        tempSample = data(:,i)-mean(data(2000:2500,i));

        step_func = fittype('a / (1 + exp(b * (x - c))) + d','independent', 'x', ...
            'coefficients', {'a', 'b', 'c', 'd'});
        
        % initial guess matters [a b c d]
        % a: amplitude (top value)
        % b: steepness of step
        % c: midpoint
        % d: offet (floor value, should be 0)

        %initialGuesses = [0.009, 2.5e6, 1e-5, 0]; % good for P61_11mu0
        %initialGuesses = [0.007, 2.5e6, 1.3e-5, 0]; %good for P61_11mu1
        %initialGuesses = [0.004, 2.0e6, 5e-6, 0]; % good for P61_11mu2
        initialGuesses = [0.003, 2.5e6, 1.25e-5, 0];

        ft = fit(time, tempSample, step_func, 'StartPoint', initialGuesses);
        fitted = feval(ft,highResTime); 

        % figure();
        % plot(highResTime,fitted,'LineWidth',1.5)
        % hold on
        % plot(time, tempSample);
        % xlabel('time (s)')
        % ylabel('current (A)')
        % legend('fit','raw')
        % axis tight
        % grid on

        tempMax = max(fitted)*(1-one_e);
        tempMin = max(fitted)*one_e;

        [~,topPos]=min(abs(fitted-tempMax));
        [~,botPos]=min(abs(fitted-tempMin));

        if botPos < topPos
            disp('Error: bottom occurs before top')
            disp(i);
        end

        % debug plot
        % don't forget to uncomment figure() outside of loop
        % plot(ft,time,tempSample)
        % hold on
        % xline(highResTime(botPos))
        % hold on
        % xline(highResTime(topPos))
        % xlabel('time')
        % ylabel('voltage')
        % pause(1)
        % clf

        decayTimes=highResTime(botPos)-highResTime(topPos);
        y_axis(i) = d2./decayTimes;
    end
end




