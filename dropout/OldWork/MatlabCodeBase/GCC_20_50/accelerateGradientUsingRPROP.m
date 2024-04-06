
% =========================================================== %
%               Backpropagation Through Time                  %
%              for Vector Control Application                 %
%                    Data Initialization                      %
%                       (October 2011)                        %
% =========================================================== %

function resultDeltaZ=accelerateGradientUsingRPROP(dRdZ,previousDeltaZ)

  % this is a simplified version of RPROP by Michael Fairbank, 
  % that is not identical to Reidmiller's paper, but captures the essence of it, and is simpler, but I've emprically found it to be just as effective.

  % RPROP constants (these are not the standard values, but should be fine):
    initial_update_value=0.01;      % Reidmiller uses 0.1 but I prefer something smaller
    eta_minus=0.5;                  % learning braking rate
    eta_plus=1.2;                   % learning acceleration rate
    max_update_value = 50;          % This is ridiculously large, but in practice (hopefully) it is never needed.
    min_update_value = 1e-8;        % Riedmillar might use something bigger here, I think.
    
    [numRows,numCols]=size(dRdZ);
    
    sign_dRdZ = sign(dRdZ);
    signPrevUpdate = sign(previousDeltaZ);
    signChange=sign_dRdZ.*signPrevUpdate;
    updateValue=abs(previousDeltaZ);
      
    for row=1:1:numRows
        for col=1:1:numCols		
            if signChange(row,col) > 0
                updateValue(row,col)=updateValue(row,col)*eta_plus;    % grow this learning rate (e.g. by a factor 1.2)                
            elseif signChange(row,col) < 0
                updateValue(row,col)=updateValue(row,col)*eta_minus;   % shrink this learning rate (e.g. by a factor 0.5)
            end
            
            if updateValue(row,col)==0
                updateValue(row,col)=initial_update_value;
            elseif updateValue(row,col) > max_update_value
                updateValue(row,col) = max_update_value;
            elseif updateValue(row,col) < min_update_value
                updateValue(row,col) = min_update_value;     % don't let learning stop completely           
            end
        end
    end
    
    resultDeltaZ = sign_dRdZ .* updateValue;  

end