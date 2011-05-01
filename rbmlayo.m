% rbm layer object
% based on Salakhutdinov's implementation
% do whatever you want with it licenc
% Mate Toth, argonzulu@gmail.com

classdef rbmlayo
    properties
        % layer parameters
        w;                              % visible X hidden
        a;                              % visible bias - row vector
        b;                              % hidden bias - row vector
        
        % layer parameter stats
        dw0;                            % momentum gradients
        da0;
        db0;
        qs;                             % variable to track node activity
        
        
        % activation funcs  data,w,b -> [ss,ps]
        vtoh;                           % visible to hidden function
        htov;
        
        % learning parameters
        eta;                            % learning rate (when using fixed iteration)
        mom;                            % learning momentum
        
        % weight threshold, penalty
        wdeq;                           % weight decay applied?
        wdet;                           % weight decay - threshold - [0.8]
        wdep;                           % weight decay - penalty - [0.1, 0.001]
       
        % sparsity target, penalty, lambda
        sprq;                           % sparsity applied?
        sprt;                           % sparsity target - [0.01, 0.1]
        sprp;                           % sparsity penalty factor - [0.1, 0.9]
        sprl;                           % sparsity lambda (updated factor) - [0.9, 0.99]        
                      
        % batch
        batchsize;
                
        % verbosity
        verbose;                        % print out free energies and changes

        
        % example: representation a 20 dimension data with 40 bernoulli binary units
        % rl=rbmlayero(20,'BB',40,'BB', ...
        %             {'sprt',0.05,'pretrainc',250,'trainc',500,'batchsize',10,'validsize',2000,'eta',0.01});
        % rl=rl.cd1train(xs)
        % rl.hidden_probs(xs)
        % rl.hidden_states(xs)
        
        % example: representation with gaussian units:
        % xs is centered
        % rl=rbmlayero(20,'Gzu',40,'BB', ...
        %             {'sprt',0.05,'pretrainc',250,'trainc',500,'batchsize',10,'validsize',2000,'eta',0.01});
        % rl=rl.cd1train(xs)
        

    end
    methods        
        function o=rbmlayo(nvis,typvis,nhid,typhid,opta)
            o.w=(rand(nvis,nhid)/100) .* sign(rand(nvis,nhid)-0.5);
            o.a= -rand(1,nvis)/100;     % because the minus bias, they are tend to start off
            o.b= -rand(1,nhid)/100;

            
            % parameters                - should be experimenting with and changed!     
            o.eta=0.1;
            o.mom=0.5;
            o.batchsize=10;
            
            o.wdeq=false;
            o.sprq=false;
            o.verbose=0;                % print out 
 
            
            % visible side
            if strcmp(typvis,'BB')
                o.htov=@logistic_state;
            elseif strcmp(typvis,'Gzu')
                o.htov=@gaussian_state;
                o.eta=0.02;
            end
 
            % hidden side
            if strcmp(typhid,'BB')
                o.vtoh=@logistic_state;
            elseif strcmp(typhid,'Gzu')
                o.vtoh=@gaussian_state;
                o.eta=0.02;
            end
           
            % supplied parameters
            if exist('opta')
                i=1;
                while i<=length(opta)
                    if strcmp(opta{i},'eta')
                        o.eta=opta{i+1};
                    elseif strcmp(opta{i},'mom')
                        o.mom=opta{i+1};
                    elseif strcmp(opta{i},'batchsize')
                        o.batchsize=opta{i+1};               
                    elseif strcmp(opta{i},'wdet')
                        o.wdet=opta{i+1};
                    elseif strcmp(opta{i},'wdep')
                        o.wdep=opta{i+1};
                    elseif strcmp(opta{i},'sprl')
                        o.sprl=opta{i+1};
                    elseif strcmp(opta{i},'sprt')
                        o.sprt=opta{i+1};
                    elseif strcmp(opta{i},'sprp')
                        o.sprp=opta{i+1};
                    elseif strcmp(opta{i},'verbose')
                        o.verbose=opta{i+1};
                    else
                        display(sprintf('WRONG PARAMETER LABEL: %s',opta{i}));
                    end
                    i=i+2;
                end
            end
            
            % setting weight decay
            if o.wdep && o.wdet
                o.wdeq=true;
                display('weight decay applied');
            end
            
            % setting sparsity
            if o.sprt && o.sprp && o.sprl
                o.sprq=true;
                o.qs=zeros(1,nhid);
                display('sparsity applied');
            end            
        end
        
        function [c]=clone(o)
        % Instantiate new object of the same class.
            c=feval(class(o));
 
            % Copy all non-hidden properties.
            ps=properties(o);
            for i=1:length(ps)
                c.(ps{i}) = o.(ps{i});
            end
        end
        
        function [o]=eta_decay(o,p)
            o.eta=o.eta*p;
            o.dw0=o.dw0*p;
            o.da0=o.da0*p;
            o.db0=o.db0*p;
            if o.verbose
                display(sprintf('old eta: %f  -  new eta: %f',o.eta/p,o.eta));
            end
        end
      
            
% batches for training
        function [bxa]=training_batches(o,xs)
            cis=crossvalind('Kfold',size(xs,1),o.batchsize);
            uis=unique(cis);
            bxa={};
            for i=1:size(uis,1)
               xs_=xs(find(cis==uis(i)),:);
               bxa=cat(1,bxa,xs_);
            end
        end

       
% monitoring
        function [e]=free_energy(o,xs)
        % the easy equation from hinton's practical guide expr(24)
        % http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
            numcases=size(xs,1);
            numhid=size(w,2);
            
            ea=xs*o.a';
            hs=xs*o.w + repmat(b,numcases,1);         
            e= (-sum(sum(ea)) - sum(sum(log(1+exp(hs))))) / numcases;
        end
        function [e,de]=free_energy_diff(o,xs,e0)
            e=o.free_energy(xs);
            de=e0-e;
        end
        function [e]=reconstruction_error(o,xs)
            [h0s,h0ps]=o.vtoh(xs,w,b);
            [v1s,v1ps]=o.htov(h0ps,w',a);
            e=sum(sum(abs((xs-v1ps).^2)))/prod(size(xs));
        end
            
        function []=print_energy(o,e,re)
            display(sprintf('eta:%f  -  energy: %f  -  recerror: %f',o.eta,e,re));
        end        
        function [e,re]=monitor_energy(o,xs,e0)
        % monitor
            e=o.free_energy_diff(xs);
            re=o.reconstruction_error(xs);
            if o.verbose
                o.print_energy(e,re);
            end
        end            
        
        
% learning
       function [dw,da,db]=learning_diff(o,dw,da,db) % diff - momentum
            dw=dw*o.eta + o.dw0*o.mom;
            da=da*o.eta + o.da0*o.mom;
            db=db*o.eta + o.db0*o.mom;
        end
        function [o]=apply_learning_diff(o,dw,da,db)
           o.w=o.w+dw; 
           o.a=o.a+da; 
           o.b=o.b+db; 
           
           o.dw0=dw;
           o.da0=da;
           o.ba0=ba;
       end
       
        function [dw]=weight_penalty(o)
            dw=((o.w-o.wdet).^2) * (o.wdep*-1*0.5*o.eta);     % WRONG COMPUTATIONALLY INEFFICIENT         
%             wp=(sum(w.^2,1))*(-1*0.5*o.wdep*o.eta);    % LEARNING RATE INCLUDED!!!
%             dw=repmat(wp,size(w,1),1);
        end        
        function [dw,db,qs]=sparsity_penalty(o,hs)
            qs=o.qs*o.sprl + hs*(1-o.sprl);
            ps=(o.sprt-qs)*(o.sprp*o.eta);             % LEARNING RATE INCLUDED!!!

            dw=repmat(ps,size(o.w,1),1);
            db=ps;
        end
        
        function [o]=cd1(o,xs)
            [ncases,ndims]=size(xs);

            %go up down up
            [h0s,h0ps]=o.vtoh(xs,o.w,o.b);
            [v1s,v1ps]=o.htov(h0s,o.w',o.a);
            [h1s,h1ps]=o.vtoh(v1ps,o.w,o.b);   

            %the learning signal;  
            dw=(v0s'*h0ps - v1s'*h1ps)/ncases; 
            da=(sum(v0s,1)-sum(v1s,1))/ncases;
            db=(sum(h0ps,1)-sum(h1ps,1))/ncases;
            
            % weight decay penalty
            if o.wdeq
                dw_wpen=o.weight_penalty();
                dw=dw+dw_wpen;
            end

            % weight sparsity penalty
            if o.sprq
                [dw_spen,db_spen,qs]=o.sparsity_penalty(mean(h0s,1));
                dw=dw+dw_spen;
                db=db+db_spen;
                o.qs=qs;
            end
            
            % learning diff
            [dw,da,db]=o.learning_diff(dw,da,db);
            o=o.apply_learning_diff(dw,da,db);     
        end       
   
        function [o]=cd1_batches(o,bxa)
            for i=1:length(bxa)
                o=o.cd1(bxa{i});
            end
        end
        function [o]=cd1_batches_ntimes(o,xs,n)
            for i=1:n
                bxa=o.training_batches(xs);    % new batches set every time - don't know is it worth it
                o=o.cd1_batches(bxa);
            end
        end
        
        function [o0,es,res]=cd1_while_energy_improve(o,xs,monitori)
        % es - energies,  res - reconstruction errors
            o0=o.clone();
            [e0,re0]=o.monitor_energy(xs);
            es=[];
            res=[];

            o1=o0.cd1_batches_ntimes(xs,monitori);
            [e1,re1]=o1.monitor_energy(xs);
            while e0<e1
               o0=o1;
               es=cat(1,es,e1);         % stats for return
               res=cat(1,res,re1);
               
               o1=o0.cd1_batches_ntimes(xs,monitori);
               [e1,re1]=o1.monitor_energy(xs);
            end            
        end
        function [o,es,res]=cd1_etadecay(o,xs,monitori,eta0,etap,etam)
            o.eta=eta0;
            es=[];
            res=[];
            while o.eta>etam
               [o,es_,res_]=o.cd1_while_energy_improve(xs,monitori);
               o=o.eta_decay(etap);
               es=cat(1,es,es_);
               res=cat(1,res,res_);
            end
        end
    
        
        function ps=hidden_probs(o,xs)
            [ss,ps]=o.vtoh(xs,o.w,o.b);
        end
        function ss=hidden_states(o,xs)
            [ss,ps]=o.vtoh(xs,o.w,o.b);
        end
        
        function v1ps=reconstruction_probs(o,xs)
            [h0s,h0ps]=o.vtoh(xs,o.w,o.b);
            [v1s,v1ps]=o.htov(h0ps,o.w',o.a);
        end
            
    end
end