function [u,v] = project_point(Trw, K, Xw)
    Xw = [Xw; 1];
    Xr = Trw*Xw;
    Xr = Xr(1:3);

    Xr(3) = abs(Xr(3));

    U = K*Xr;
    u = U(1)/U(3);
    v = U(2)/U(3);
end


