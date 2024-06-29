% jv is 4-D dlarray
% tempPred is 1xn single
% material_params is 1xn single

function [mapeError, mseError, maeError, rmsleError, rsError] = forwardLoss(inputNet,jv,material_params) 

    tempPred=extractdata(predict(inputNet,jv));
    
    % use below to check one push using predict gives same as loop
    % doubleJV = extractdata(jv);
    % pred = zeros(size(material_params));
    % for i = 1:length(material_params)
    %     pred(i) = forward(inputNet,dlarray(doubleJV(:,:,:,i),'SSCB'));
    % end

    % Mean Absolute Percentage Error (MAPE)
    tempLoss=abs((material_params-tempPred)./material_params);
    sumLoss=sum(tempLoss(isfinite(tempLoss)));
    mapeError=sumLoss/sum(isfinite(tempLoss));

    % Mean squared error (MSE)
    mseError = immse(tempPred,single(material_params));
    
    % MAE
    maeError = sum(abs(tempPred-material_params))/length(tempPred);

    % RMSLE
    rmsleError = sqrt(sum((material_params-tempPred).^2)/length(tempPred));

    % R-squared
    % https://en.wikipedia.org/wiki/Coefficient_of_determination
    sst = sum((material_params-mean(material_params)).^2);
    ssr = sum((material_params-tempPred).^2);
    rsError = 1 - ssr/sst;

    % Adjusted R-squared
    % ?

end