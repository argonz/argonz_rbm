% rbm layer object
% based on Salakhutdinov's implementation
% do whatever you want with it licenc
% Mate Toth, argonzulu@gmail.com


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
        etadecq;                        % using eta decay?
        eta0;                           % starting eta parameter
        etap;                           % eta decay parameter: eta_new = eta_old * etap;
        etam;                           % minimum eta threshold - stop the algorithm
        etai;                           % eta check frequency (check energy still improve)
        
        eta;                            % learning rate (when using fixed iteration)
        mom;                            % learning momentum
        
        % statistics - for measurement
        etahist;                        % eta values history
        trehist;                        % free energies on training set
        vaehist;                        % on validation set
        trrhist;                        % reconstruction error on training set
        varhist;                        % on validation set


        % weight threshold, penalty
        wdet;                           % weight decay - threshold - [0.8]
        wdep;                           % weight decay - penalty - [0.1, 0.001]
       
        % sparsity target, penalty, lambda
        sprt;                           % sparsity target - [0.01, 0.1]
        sprp;                           % sparsity penalty factor - [0.1, 0.9]
        sprl;                           % sparsity lambda (updated factor) - [0.9, 0.99]
              
        % batch
        batchsize;
        
        % validation set parameters
        validsetq;                      % use validation set for monitoring energy
        detervalq;                      % use non random sample as validation set (for measurement)
        validsize;                      % size of the validation set for free energy
        
        
        % overfitting related
        fixediterq                      % using fixed iteration
        iter                            % number of fixed iteration
        
        
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
            o.a= -rand(1,nvis)/100;     % because the minus bias, they are tend to start off
            o.b= -rand(1,nhid)/100;

            
            % parameters                - should be experimenting with and changed!     
            o.eta0=0.2;
            o.etam=0.002;
            o.etap=0.8;
                            
            o.mom=0.5;
            o.batchsize=10;
            
            o.validsetq=false;
            o.validsize=100;            % should be arount 1/20 of the data? :O
            
            o.wdet=0.8;
            o.wdep=0.1;             % values from the manual :)    
            
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
                    elseif strcmp(opta{i},'validsetq')
                        o.validsetq=opta{i+1};
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
         
% random validation set - vxs: validation - txs: training - cxs: compare from training
        function [vxs,txs,cxs]=exclude_rand_validset(o,xs)
            is=1:size(xs,1);
            vis=randsample(is,o.validsize);
            tis=setdiff(is,vis);
            cis=randsample(tis,o.validsize);
            
            vxs=xs(vis,:);
            cxs=xs(cis,:);
            txs=xs(vtis,:);
        end
        function [vxs,txs,cxs]=exclude_freq_validset(o,xs)  
            len=size(xs,1)
            is=1:len;
            vis=1:floor(len/o.validsize):len;
            tis=setdiff(is,vis);
            
            len=size(tis,1)
            cis=vis(1:floor(len/o.validsize):len,:);
            
            vxs=xs(vis,:);
            txs=xs(tis,:);
            cxs=xs(cis,:);
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
        function [e]=free_energy(o,xs,w,a,b)
        % the easy equation from hinton's practical guide expr(24)
        % http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
            numcases=size(xs,1);
            numhid=size(w,2);
            
            ea=xs*a';
            hs=xs*w + repmat(b,numcases,1);         
            e= (-sum(sum(ea)) - sum(sum(log(1+exp(hs))))) / numcases;
        end
        function [e,de]=free_energy_diff(o,xs,e0,w,a,b)
            e=o.free_energy(xs,w,a,b);
            de=e0-e;
        end
        function [ev,et]=free_energies(o,vxs,txs,w,a,b)
            ev=o.free_energy(vxs,w,a,b);
            et=o.free_energy(txs,w,a,b);
        end  
        function [e]=reconstruction_error(o,v0s,w,a,b)
            [h0s,h0ps]=o.vtoh(v0s,w,b);
            [v1s,v1ps]=o.htov(h0ps,w',a);
            e=sum(sum(abs((v0s-v1ps).^2)))/prod(size(v0s));
        end
            
        function []=print_energy_state(o,e,re)
            display(sprintf('eta:%f  -  energy: %f  -  recerror: %f',o.eta,e,re));
        end
        function []=print_val_trn_energy_state(o,vale,valre,trne,trnre)
            display(sprintf('eta:%f  -  ve: %f / te: %f  -  vre: %f / tre: %f',o.eta,vale,trne,valre,trnre));
        end
        
        function [de,e,re]=monitor_energy_state(o,xs,w,a,b,e0,i)
        % monitor
            e=o.free_energy(xs,e0,w,a,b);
            re=o.reconstruction_error(xs,w,a,b);
            de=e-e0;
            if o.verbose
                o.print_energy_state(i,e,re);
            end
        end            
        function [vde,ve,vre,tde,te,tre]=monitor_val_trn_energy_state(o,vxs,txs,w,a,b,ve0,te0,i)
                   
            ve=o.free_energy(vxs,e0,w,a,b);
            vre=o.reconstruction_error(vxs,w,a,b);
            vde=ve-ve0;
                
            te=o.free_energy(txs,e0,w,a,b);
            tre=o.reconstruction_error(txs,w,a,b);
            tde=te-te0;
                
            if o.verbose
                o.print_val_trn_energy_state(i,ve,vre,te,tre);
            end
        end
        
        
% learning
        function [dw]=weight_penalty(o,w)
            dw=((w-o.wdet).^2) * (o.wdep*-1*o.eta);     % WRONG COMPUTATIONALLY INEFFICIENT         
%             wp=(sum(w.^2,1))*(-1*0.5*o.wdep*o.eta);    % LEARNING RATE INCLUDED!!!
%             dw=repmat(wp,size(w,1),1);
        end        
        function [dw,db,qs]=sparsity_penalty(o,w,b,hs,qs)
            qs=qs*o.sprl + hs*(1-o.sprl);
            ps=(o.sprt-qs)*(o.sprp*o.eta);             % LEARNING RATE INCLUDED!!!

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
            [h1s,h1ps]=o.vtoh(v1ps,w,b);   

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
            
%              dw=dw+dw_wpen+dw_spen;
             dw=dw+dw_spen;
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
        function [w,a,b,dw,da,db,qs]=cd1train_allbatches_ntimes(o,n,xs,w,a,b,dw0,da0,db0,qs)
            for i=1:n
                bxa=o.training_batches(txs);    % new batches set every time - don't know is it worth it
                [w,a,b,dw,da,db,qs]=cd1train_allbatches(bxa,w,a,b,dw0,da0,db0,qs);
            end
        end
        
        function ar=save_train_state(o,w,a,b,dw0,da0,db0,qs)
            ar={w,a,b,dw0,da0,db0,qs};
        end
        function [w,a,b,dw0,da0,db0,qs]=load_train_state(o,ar)
            w=ar{1};
            a=ar{2};
            b=ar{3};
            dw0=ar{4};
            da0=ar{5};
            db0=ar{6};
            qs=ar{7};
        end
        function [dw0,da0,db0]=decay_train_state(o,dw0,da0,db0,p)
            dw0=dw0*p;
            da0=da0*p;
            db0=db0*p;
        end
        function [w,a,b,dw,da,db,qs]=cd1train_till_energy_improve(o,xs,w,a,b,dw0,da0,db0,qs)
            ar=o.save_train_state(w,a,b,dw0,da0,db0,qs);
            [de,e,re]=monitor_energy_state(o,xs,w,a,b,e0);
            
            [w,a,b,dw,da,db,qs]=o.cd1train_allbatches_ntimes(o.etai,xs,w,a,b,dw0,da0,db0,qs);
            o.cd1train_allbatches_ntimes
            
        end
       
       function [o]=cd1train(o,xs)
            
            % if use validset and random sampling
            if o.validsetq && ~detervalq; [vxs,txs,cxs]=o.exclude_validset(xs); end
            if o.validsetq && detervalq; [vxs,txs,cxs]=o.exclude_validset(xs); end
            
            % parameters 
            w=o.w;
            a=o.a;
            b=o.b;
            dw0=0;
            db0=0;
            da0=0;
           
            qs=zeros(1,size(w,2));       
            
            % different kind of trainings
            if ~o.validsetq             
                o.eta=o.eta0;
                [de,e,re]=monitor_energy_state(o,xs,w,a,b,e0,i);
                                
                while o.eta>o.etam;
                    i=0;
                    dev=1;
                    
                    [w,a,b,dw,da,db,qs]=o.cd1train_allbatches_ntimes(o.etai,xs,w,a,b,dw0,da0,db0,qs);
                    [de,e,re]=monitor_energy_state(o,xs,w,a,b,e0,i);
                    
                    [ev,dev]=o.free_energy_diff(xs,0,w,a,b);
                    er=o.reconstruction_error(xs,w,a,b);
                    o.print_energy_state(i,ev,er);

                end
            else
                
            end

            
            % pre training with maximum eta
            [ev,et,dev,det]=o.free_energies(vxs,vtxs,0,0,w,a,b);
            for c=0:o.pretrainc
                bxa=o.training_batches(txs);    % new batches set every time - don't know is it worth it
                [w,a,b,dw,da,db,qs]=o.cd1train_allbatches(bxa,w,a,b,dw0,da0,db0,qs);

                % status report
                if o.verbose && mod(c,modp)==0                    
                    % energy changes
                    [ev,et,dev,det]=o.free_energies(vxs,vtxs,ev,et,w,a,b);
                    evr=o.reconstruction_error(vxs,w,a,b);
                    etr=o.reconstruction_error(vtxs,w,a,b);
                    display(sprintf('pretrain: %d :: free energies  v:%f - t:%f :: recons err  v:%f - t:%f',c,ev,et,evr,etr));
                end                 
            end
   

            % training with decaying eta and a slight overfit watch
            c=0;
            while c<o.trainc      % || (mdet/mdev)<1.25          % overfit criteria - not working, bogus
                % training
                bxa=o.training_batches(txs);    % new batches set every time - don't know is it worth it
                [w,a,b,dw,da,db,qs]=o.cd1train_allbatches(bxa,w,a,b,dw0,da0,db0,qs);

                % status report
                if o.verbose && mod(c,modp)==0                    
                    % energy derivatives
                    [ev,et,dev,det]=o.free_energies(vxs,vtxs,ev,et,w,a,b);
                    evr=o.reconstruction_error(vxs,w,a,b);
                    etr=o.reconstruction_error(vtxs,w,a,b);
                    display(sprintf('train: %d :: free energies  v:%f - t:%f :: recons err  v:%f - t:%f',c,ev,et,evr,etr));
                end

                o.eta=o.eta*(o.trainc/(o.trainc-c))*((o.trainc-c-1)/o.trainc);  
                c=c+1;                              
            end 
            
            % print activity 
            display(sprintf('activity statistics   mean: %f  sd: %f',mean(qs),std(qs)));

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
        
        function v1ps=reconstruction_probs(o,xs)
            [h0s,h0ps]=o.vtoh(xs,o.w,o.b);
            [v1s,v1ps]=o.htov(h0ps,o.w',o.a);
        end
            
    end
end