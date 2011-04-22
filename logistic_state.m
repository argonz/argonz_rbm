        function [ss,ps]=logistic_state(xs,w,b)
            numcases=size(xs,1);
            numhid=size(w,2);
            ps=logistic(xs*w + repmat(b,numcases,1));	% activation
            ss=ps > rand(numcases,numhid);
        end