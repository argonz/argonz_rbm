% rbm layer object
% here and there based on Salakhutdinov's original implementation
% do whatever you want with it licenc
% Mate Toth : argonzulu@gmail.com
classdef rbmdbno
    properties
        rlos;
        
        % example: 3 layer, 20 dimension data, 50 hidden unit, first gaussian, then bernoulli representation 
        % rdo=rbmdbno();
        % rdo=rdo.add_layer(20,'Guz',50,'BB',{'eta',0.005,'sprt',0.1});
        % rdo=rdo.add_layer_n(2,50,'BB',50,'BB',{'eta',0.1,'sprt',0.1});
        % rdo=rdo.cd1trains(xs); 
        
        % rdo=rbmdbno({20,'Guz',50,'BB',{'eta',0.005,'sprt',0.1}}, ...
        %             {50,'BB',50,'BB',{sprt',0.1}}, ...
        %             {50,'BB',50,'BB',{sprt',0.1}}};
        
% $$$         rdo=rbmdbno()
% $$$         rdo=rdo.add_layer(16,'Guz',50,'BB',{'eta',0.005,'sprt',0.1,'pretrainc',50,'trainc',100});
% $$$         rdo=rdo.add_layer_n(2,50,'BB',50,'BB',{'eta',0.1,'sprt',0.1,'pretrainc',50,'trainc',100})
        
% $$$         {16,'Guz',50,'BB',{'eta',0.005,'sprt',0.1}}, ...
        %             {50,'BB',50,'BB',{sprt',0.1}}, ...
        %             {50,'BB',50,'BB',{sprt',0.1}}};
        
    end
    methods
        function o=rbmdbno()
            o.rlos={};
        end
        function o=add_layer(o,nvis,typvis,nhid,typhid,opta)
                rlo=rbmlayero(nvis,typvis,nhid,typhid,opta);
                o.rlos=cat(2,o.rlos,{rlo});
        end
        function o=add_layer_n(o,n,nvis,typvis,nhid,typhid,opta)
                for i=1:n
                    o=o.add_layer(nvis,typvis,nhid,typhid,opta);
                end
        end
        function o=cd1trains(o,xs)
           for i=1:length(o.rlos)
               rlo=o.rlos{i};
               rlo=rlo.cd1train(xs);
               xs=rlo.hidden_probs(xs);
               o.rlos{i}=rlo;
           end
       end
       function ps=prob_repr_nth(o,xs,n)
           for i=1:length(o.rlos)
               rlo=o.rlos{i};
               xs=rlo.hidden_probs(xs);
           end
           ps=xs;
       end
       function ps=prob_repr(o,xs)
           ps=o.prob_repr_nth(xs,length(o.rlos));
       end
        
    end
end
