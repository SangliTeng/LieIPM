classdef MultiRigidBody3D
    properties (Access = public)
        num_srb = 0;
        mrb = [];
        dt = 0;
    end

    methods
        function obj = MultiRigidBody3D(num_srb, srbs, dt)
            obj.mrb = srbs;
            obj.num_srb = length(srbs);
            obj.dt  = dt;
        end

        function obj = BatchRetractSO3xR3(obj, dx)
            dx = reshape(dx, [12, obj.num_srb]);
            mrb_temp = obj.mrb;
            for k = 1:obj.num_srb
                 mrb_temp(k) = mrb_temp(k).RetractSO3xR3(dx(:, k));   
            end
            obj.mrb = mrb_temp;
        end

        function xvec = GetVectorizedState(obj)
            xvec = [];
            for k = 1:obj.num_srb
                xvec = [xvec;
                        obj.mrb(k).R(:);
                        obj.mrb(k).p(:);
                        obj.mrb(k).F(:);
                        obj.mrb(k).v(:)];
            end
        end
    end
end