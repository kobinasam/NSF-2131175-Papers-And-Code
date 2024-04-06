
% =========================================================== %
%          Backpropagation Through Time for Vector            %
%                    Control Application                      %
%             This function specifies how idq_ref changes over time, (to make the problem more difficult)        %
%    It is not actually used yet (as at 6-dec-2011).          %
%                       (October 2011)                        %
% =========================================================== %


function idq_ref_new = modifyIdq_ref(idq_ref) 
	% on entry, timestep is the timestep of the trajectory (integer), and trajectoryNumber is the number of this trajectory (this will provide a random number seed).

    global Vdq Vmax XL;

    Vd=Vdq(1); id_ref=idq_ref(1); iq_ref=idq_ref(2);
    
    Vq1=id_ref*XL; Vd1=sqrt(Vmax^2-Vq1^2);
    iq_ref_new=(Vd1-Vd)/XL;
    
    if (iq_ref>iq_ref_new)
        iq_ref=iq_ref_new;
    end
    
    idq_ref_new=[id_ref; iq_ref];
 
end

