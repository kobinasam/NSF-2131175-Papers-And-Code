
% =========================================================== %
%          Backpropagation Through Time for Vector            %
%                    Control Application                      %
%             This function specifies how idq_ref changes over time, (to make the problem more difficult)        %
%    It is not actually used yet (as at 6-dec-2011).          %
%                       (October 2011)                        %
% =========================================================== %


function idq_ref = calculateIdq_ref(trajectoryNumber,timeStep) 
	% on entry, timestep is the timestep of the trajectory (integer), and trajectoryNumber is the number of this trajectory (this will provide a random number seed).

    global Ts Imax Iq_max;

    idq_ref=[Imax/2; 0];
    periodToChangeIdqRef=0.1;  %This is the time interval that specifies how often idq_ref is to change (in seconds);
    changeNumber=floor((timeStep-1)*Ts/periodToChangeIdqRef);  % use floor here?
%     timeStep, Ts
%     pause;
    
    if (changeNumber>0) 
       seed_num=trajectoryNumber*10000+changeNumber;
%        RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed_num));
       rand ('state', seed_num);
%        idq_ref=[100;0]+50*randn(2,1);
       idq_ref=[-Imax;-Imax]+[2*Imax;Iq_max+Imax].*rand(2,1);
       idq_ref=modifyIdq_ref(idq_ref);
    end
end
