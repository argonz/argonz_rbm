function [ys]=logistic(xs)
    ys=1./(1.+exp(-xs*2));
end
