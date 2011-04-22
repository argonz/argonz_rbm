% rbm layer object
% here and there based on Salakhutdinov's original implementation
% do whatever you want with it licenc
% Mate Toth : argonzulu@gmail.com


classdef rbmlayero
    properties
        % layer parameters
        w;                              % visible X hidden
        a;                              % visible bias - row vector
        b;                              % hidden bias - row vector

        % activation funcs  data,w,b -> [ss,ps]
        vtoh;                           % visible to hidden function
        htov;
        
        % learning parameters
        eta;                            % learning rate
        mom;                            % learning momentum

        wdec;                           % weight decay - for exploding weights - [0.001, 0.00001]
        sprp;                           % sparsity penalty factor - [0.1, 0.9]
        sprt;                           % sparsity target - [0.01, 0.1]
        sprl;                           % sparsity lambda (updated factor) - [0.9, 0.99]
       
        % batch
        batchsize;
        validsize;                      % validation set for free energy
        
        % overfitting related
        pretrainc;                      % number of epochs with eta
        trainc;                         % number of epochs with lowering eta
        
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
        function o=rbmlayero(nvis,typvis,nhid,typhid,opta)
            o.w=(rand(nvis,nhid)/100) .* sign(rand(nvis,nhid)-0.5);
            o.a=(rand(1,nvis)/100 .* sign(rand(1,nvis)-0.5));
            o.b=(rand(1,nhid)/100 .* sign(rand(1,nhid)-0.5));

            
            % parameters                - should be experimenting and changed!     
            o.eta=0.1;
            o.mom=0.5;
            o.batchsize=10;
            o.validsize=100;            % should be arount 1/20 of the data? :O
            
            o.wdec=0.00001;             % values from the manual :)            
            o.sprt=0.04;                % should be between 0.01 - 0.1
            o.sprp=0.4;                 % sparsity penalty cos
            o.sprl=0.95;                % sparsity lambda - tracking activity statistics

            o.pretrainc=20;             % pretrain count - static eta
            o.trainc=30;                % train count - lowering eta
            
            o.verbose=1;                % print out 
 
            
            % visible side
            if strcmp(typvis,'BB')
                o.htov=@logistic_state;
            elseif strcmp(typvis,'Gzu')
                o.htov=@gaussian_state;
                o.eta=0.005;
            end
 
            % hidden side
            if strcmp(typhid,'BB')
                o.vtoh=@logistic_state;
            elseif strcmp(typhid,'Gzu')
                o.vtoh=@gaussian_state;
                o.eta=0.005;
            end
           
            % supplied parameters
            if exist('opta')
                i=1;
                while i<length(opta)
                    if strcmp(opta{i},'eta')
                        o.eta=opta{i+1};
                    elseif strcmp(opta{i},'momentum')
                        o.mom=opta{i+1};
                    elseif strcmp(opta{i},'batchsize')
                        o.batchsize=opta{i+1};
                    elseif strcmp(opta{i},'validsize')
                        o.validsize=opta{i+1};
                    elseif strcmp(opta{i},'sprl')
                        o.sprl=opta{i+1};
                    elseif strcmp(opta{i},'sprt')
                        o.sprt=opta{i+1};
                    elseif strcmp(opta{i},'sprp')
                        o.sprp=opta{i+1};
                    elseif strcmp(opta{i},'pretrainc')
                        o.pretrainc=opta{i+1};
                    elseif strcmp(opta{i},'trainc')
                        o.trainc=opta{i+1};
                    elseif strcmp(opta{i},'verbose')
                        o.verbose=opta{i+1};
                    else
                        display(sprintf('WRONG PARAMETER LABEL: %s',opta{i}));
                    end
                    i=i+2;
                end
            end
            
        end
         
        function [vxs,vtxs,txs]=exclude_validset(o,xs)
            is=1:size(xs,1);
            vis=randsample(is,o.validsize);
            tis=setdiff(is,vis);
            vtis=randsample(tis,o.validsize);
            
            vxs=xs(vis,:);
            txs=xs(tis,:);
            vtxs=xs(vtis,:);
        end
        function [bxa]=training_batches(o,xs)
            cis=crossvalind('Kfold',size(xs,1),o.batchsize);
            uis=unique(cis);
            bxa={};
            for i=1:size(uis,1)
               xs_=xs(find(cis==uis(i)),:);
               bxa=cat(1,bxa,xs_);
            end
        end

        function [e]=free_energy(o,xs,w,a,b)
        % the easy equation from hinton's practical guide expr(24)
        % http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
            numcases=size(xs,1);
            numhid=size(w,2);
            
            ea=xs*a';
            hs=xs*w + repmat(b,numcases,1);         
            e= (-sum(sum(ea)) - sum(sum(log(1+exp(hs))))) / numcases;
        end
        function [ev,et,dev,det]=free_energies(o,vxs,txs,ev0,et0,w,a,b)
            ev=o.free_energy(vxs,w,a,b);
            et=o.free_energy(txs,w,a,b);
            dev=ev-ev0;
            det=et-et0;
        end
        function [e]=reconstruction_error(o,v0s,w,a,b)
            [h0s,h0ps]=o.vtoh(v0s,w,b);
            [v1s,v1ps]=o.htov(h0ps,w',a);
            e=sum(sum(abs((v0s-v1ps).^2)));
        end
        function [dw]=weight_penalty(o,w)
            wp=(sum(w.^2,1))*(-1*0.5*o.wdec*o.eta);    % LEARNING RATE INCLUDED!!!
            dw=repmat(wp,size(w,1),1);
        end        
        function [dw,db,qs]=sparsity_penalty(o,w,b,hs,qs)
            qs=qs*o.sprl + hs*(1-o.sprl);
            ps=(o.sprt-qs)*o.sprp;

            dw=repmat(ps,size(w,1),1);
            db=ps;
        end
        function [dw,da,db]=learning_diff(o,dw,da,db,dw0,da0,db0) % diff - momentum
            dw=dw*o.eta + dw0*o.mom;
            da=da*o.eta + da0*o.mom;
            db=db*o.eta + db0*o.mom;
        end
        function [dw,da,db,qs]=cd1(o,v0s,w,a,b,qs)
            [numcases,numdims]=size(v0s);

            %go up down up
            [h0s,h0ps]=o.vtoh(v0s,w,b);
            [v1s,v1ps]=o.htov(h0s,w',a);
            [h1s,h1ps]=o.vtoh(v1s,w,b);   

            %the learning signal;  
            dw=(v0s'*h0ps - v1s'*h1ps)/numcases; 
            da=(sum(v0s,1)-sum(v1s,1))/numcases;
            db=(sum(h0ps,1)-sum(h1ps,1))/numcases;
            
            % weight decay penalty
            dw_wpen=o.weight_penalty(w);

            % weight sparsity penalty
            [dw_spen,db_spen,qs]=o.sparsity_penalty(w,b,mean(h0s,1),qs);

            % experimenting with penalizes based on the weight
% $$$             dw_spen=((1+abs(w))/2).*dw_spen; 
% $$$             db_spen=((1+abs(b))/2).*db_spen;                                      
% $$$             dw_wpen=((1+abs(w))/2).*dw_wpen;              % weight high penalties
            
            dw=dw+dw_wpen+dw_spen;
            db=db+db_spen;           
        end       
        function [w,a,b,dw,da,db,qs]=cd1train_batch(o,xs,w,a,b,dw0,da0,db0,qs)
            [dw,da,db,qs]=o.cd1(xs,w,a,b,qs);
            [dw,da,db]=o.learning_diff(dw,da,db,dw0,da0,db0); % diff - momentum
            w=w+dw;
            a=a+da;
            b=b+db;
        end
        function [w,a,b,dw,da,db,qs]=cd1train_allbatches(o,bxa,w,a,b,dw0,da0,db0,qs)
            for i=1:length(bxa)
                [w,a,b,dw,da,db,qs]=o.cd1train_batch(bxa{i},w,a,b,dw0,da0,db0,qs);
            end
        end
        function [o]=cd1train(o,xs)
            [vxs,vtxs,txs]=o.exclude_validset(xs);
            w=o.w;
            a=o.a;
            b=o.b;
            dw0=0;
            db0=0;
            da0=0;
            qs=zeros(1,size(w,2));

            
            % pre training with maximum eta
            [ev,et,dev,det]=o.free_energies(vxs,vtxs,0,0,w,a,b);
            for c=1:o.pretrainc
                bxa=o.training_batches(txs);    % new batches set every time - don't know is it worth it
                [w,a,b,dw,da,db,qs]=o.cd1train_allbatches(bxa,w,a,b,dw0,da0,db0,qs);

                % status report
                if o.verbose && mod(c,20)==0                    
                    % energy changes
                    [ev,et,dev,det]=o.free_energies(vxs,vtxs,ev,et,w,a,b);
                    er=o.reconstruction_error(vxs,w,a,b);
% $$$                     display(sprintf('pretrain: %d :: free energies  valid:%f  -  train:%f',c,ev,et));
                    display(sprintf('pretrain: %d :: free energies  valid:%f - train:%f  :: err: %f',c,ev,et,er));
                end
                
  
            end
            
            % training with decaying eta and a slight overfit watch
            c=0;
            while c<o.trainc      % || (mdet/mdev)<1.25          % overfit criteria - not working, bogus
                % training
                bxa=o.training_batches(txs);    % new batches set every time - don't know is it worth it
                [w,a,b,dw,da,db,qs]=o.cd1train_allbatches(bxa,w,a,b,dw0,da0,db0,qs);

                % status report
                if o.verbose && mod(c,20)==0                    
                    % energy derivatives
                    [ev,et,dev,det]=o.free_energies(vxs,vtxs,ev,et,w,a,b);
                    er=o.reconstruction_error(vxs,w,a,b);
% $$$                     display(sprintf('train: %d :: free energies  valid:%f  -  train:%f',c,ev,et));
                    display(sprintf('train: %d :: free energies  valid:%f - train:%f  :: err: %f',c,ev,et,er));
                end

                o.eta=o.eta*(o.trainc/(o.trainc-c))*((o.trainc-c-1)/o.trainc);  
                c=c+1;              
            end

            % set the final parameters (slightly overfit ones)  
            o.w=w;
            o.a=a;
            o.b=b;
        end

        
        function ps=hidden_probs(o,xs)
            [ss,ps]=o.vtoh(xs,o.w,o.b);
        end
        function ss=hidden_states(o,xs)
            [ss,ps]=o.vtoh(xs,o.w,o.b);
        end
            
    end
end