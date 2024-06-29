clc
clear all
close all

%% pl

%load N57_7pl.mat
wavelengths = readmatrix('nirWavelengths.xlsx');

% Multiple

%data =  N57_7pl;
data = readmatrix('N17_6 pl.xlsx');
N17_6plA = zeros(1,size(data,2));
N17_6plW = zeros(1,size(data,2));
%figure();
for i = 1:size(data,2)
    [~,N17_6plA(i),N17_6plW(i)] = plFitMain(wavelengths,data(:,i));
end
 
%save('P58_6deltaQ2024.mat','P58_6deltaQ')

%visualizeMap(N57_7plA,'pl counts (relative)',80,0.025)
%saveas(gcf,'P58_6deltaQ map.png');

%% single sample testing
%plSample = N57_7pl(:,1568);
%plSample = plSample - mean(plSample(400:512));

%figure(); 
%plot(wavelengths,sample)

%[fitted_spectrum, fitCount, fitWavel] = plFitMain(wavelengths,plSample);

% figure();
% plot(wavelengths,fitted_spectrum)
% hold on
% plot(wavelengths,plSample)
% legend('fit','raw')
 

function [fitted_spectrum, fitCount, fitWavel] = plFitMain(wavelengths,rawData)
    
    low_range = 400:500; % change this to where you want the signal to decay to

    rawData = rawData -mean(rawData(low_range));

    % Define the Gaussian fit type
    gaussianFitType = fittype('a*exp(-((x-b)^2)/(2*c^2))', 'independent', ...
        'x', 'coefficients', {'a', 'b', 'c'});

    % Estimate initial parameters for fitting
    a0 = max(rawData); % Initial guess for the amplitude
    b0 = wavelengths(find(rawData == max(rawData), 1)); % Initial guess for the mean (center)
    
    % Estimate the width (c) from the data
    halfMax = a0 / 2;
    indicesAboveHalfMax = find(rawData >= halfMax);
    if length(indicesAboveHalfMax) > 1
        FWHM = wavelengths(indicesAboveHalfMax(end)) - wavelengths(indicesAboveHalfMax(1));
        c0 = FWHM / 2.3548; % Convert FWHM to standard deviation
    else
        c0 = std(wavelengths); % Fallback if unable to estimate FWHM
    end

    initialGuess = [a0, b0, c0];

    % Perform the fitting
    [fitresult, ~] = fit(wavelengths, rawData, gaussianFitType, 'Start', initialGuess);

    % Evaluate the fitted curve at the given wavelengths
    fitted_spectrum = feval(fitresult, wavelengths);

    % dont forget to uncomment figure before loop outside of this function
    % %figure();
    % plot(wavelengths,fitted_spectrum,'LineWidth',1.5)
    % hold on
    % plot(wavelengths,rawData)
    % legend('fit','raw')
    % xlabel('wavelengths (nm)')
    % ylabel('counts')
    % axis tight
    % grid on
    % pause(0.1)
    % clf

    [fitCount,fitIndex] = max(fitted_spectrum);
    
    fitWavel = wavelengths(fitIndex);
end

