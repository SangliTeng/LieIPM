classdef RigidBody3D
    properties (Access = public)
        id = nan;
        
        J  = eye(3);
        Jd = eye(3);
        m  = 1.0;

        R  = eye(3);
        p  = zeros(3, 1);
        F  = eye(3);
        v  = zeros(3, 1);
    end

    methods
        function obj = RigidBody3D(id, m, J)
            obj.id = id;
            obj.m  = m;
            obj.J  = J;
            obj.Jd = fun_J2Jd(J);
        end

        function obj = RetractSO3xR3(obj, dx)
            obj.R = obj.R * Rodrigues(dx(1:3));
            obj.p = obj.p + dx(4:6);
            obj.F = obj.F * Rodrigues(dx(7:9));
            obj.v = obj.v + dx(10:12);
        end

        function x = getQuaternionState(obj)
            x = struct();
            x.q = rotm2quat(obj.R);
            x.p = obj.p;
            x.w = rotm2quat(obj.F);
            x.v = obj.v;
        end
    end
end