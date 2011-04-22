function [ss,ms]=gaussian_state(xs,w,b)
    numcases=size(xs,1);
    numhid=size(w,2);
    ms=xs*w + repmat(b,numcases,1);	% activation - the mean
    ss=normrnd(ms,1);
end