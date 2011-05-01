% rbm layer object
% based on Salakhutdinov's implementation
% do whatever you want with it licenc
% Mate Toth, argonzulu@gmail.com


classdef rbmtrno
    properties
        % the rbm layer
        rbm;
                
        % statistics - for measurement
        etahist;                        % eta values history
        trehist;                        % free energies on training set
        vaehist;                        % on validation set
        trrhist;                        % reconstruction error on training set
        varhist;                        % on validation set
        
        % validation set parameters
        validsetq;                      % use validation set for monitoring energy
        detervalq;                      % use non random sample as validation set (for measurement)
        validsize;                      % size of the validation set for free energy
        
        % training set
        vxs;                            % validation set
        txs;                            % training set
        cxs;                            % compare set - subset of training set for measurement purpose
        
        % learning with eta decay
        etadecq;                        % using eta decay?
        eta0;                           % starting eta parameter
        etap;                           % eta decay parameter: eta_new = eta_old * etap;
        etam;                           % minimum eta threshold - stop the algorithm
        monitori;                       % eta check frequency (check energy still improve)
        
        % learning with fixed iteration
        iterq;                          % using fixed iteration
        iter;                           % number of fixed iteration
                
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
        function o=rbmtrno(opta)
            
            % parameters                - should be experimenting with and changed!     
            o.etap=0.8;
            o.etadecq=0;
            o.monitori=40;
                                        
            o.validsetq=0;
            o.detervalq=0;
            o.validsize=100;            % should be arount 1/20 of the data? :O
           
            o.verbose=0;                % print out 
            
            % supplied parameters
            if exist('opta')
                i=1;
                while i<=length(opta)
                    if strcmp(opta{i},'eta0')
                        o.eta0=opta{i+1};
                    elseif strcmp(opta{i},'etam')
                        o.etam=opta{i+1};
                    elseif strcmp(opta{i},'etap')
                        o.etap=opta{i+1};
                    elseif strcmp(opta{i},'monitori')
                        o.monitori=opta{i+1};
                    elseif strcmp(opta{i},'detervalq')
                        o.detervalq=opta{i+1};
                    elseif strcmp(opta{i},'validsize')
                        o.validsize=opta{i+1};
                    elseif strcmp(opta{i},'iter')
                        o.iter=opta{i+1};
                    elseif strcmp(opta{i},'verbose')
                        o.verbose=opta{i+1};
                    else
                        display(sprintf('WRONG PARAMETER LABEL: %s',opta{i}));
                    end
                    i=i+2;
                end
            end
            
            % using validation set
            if o.validsize
                o.validsetq=1;
            end
            
            % using etadecay
            if o.eta0 && o.etam && o.etap
                o.etadecq=1;
                o.iterq=0;
            end
            
            % using fixed iter
            if o.iter
                o.iterq=1;
                o.etadecq=0;
            end
        end
    
% adding rbm to the training
        function [o]=set_rbm(o,nvis,typvis,nhid,typhid,opta)
            o.rbm=rbmlayo,nvis,typvis,nhid,typhid,opta);
        end
% resetting history
        function [o]=set_hist(o)
            o.etahist=[];
            o.trehist=[];
            o.vaehist=[];
            o.trrhist=[];
            o.varhist=[];
        end
        function [o]=add_trn_hist(o,eta,es,rs)
            o.etahist=cat(1,o.etahist,repmat(eta,size(es,1),1));
            o.trehist=cat(1,o.trehist,es);
            o.trrhist=cat(1,o.trrhist,rs);
        end
        function [o]=add_trn_val_hist(o,eta,tes,trs,ves,vrs)
            o.etahist=cat(1,o.etahist,repmat(eta,size(es,1),1));
            o.trehist=cat(1,o.trehist,tes);
            o.trrhist=cat(1,o.trrhist,trs);
            o.vaehist=cat(1,o.vaehist,ves);
            o.varhist=cat(1,o.varhist,vrs);
        end
        
% resetting eta parameters
        function [o]=set_eta(o,eta0,etap,etam)
            o.eta0=eta0;
            o.etap=etap;
            o.etam=etam;
            o.etadecq=1;
            o.iterq=0;
        end
% resetting fixed iteration
        function [0]=set_iter(o,iter)
            o.iter=iter;
            o.iterq=1;
            o.etadecq=0;
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
        function [o]=set_trainingsets(o,xs)

           % if don't use validation set
           if ~o.validsetq
               o.txs=xs;
           end
           
           % if use validation set
           if o.validsetq
              if ~detervalq
                  [vxs,txs,cxs]=o.exclude_validset(xs); 
                  o.vxs=vxs;
                  o.txs=txs;
                  o.cxs=cxs;
              end
              if detervalq 
                  [vxs,txs,cxs]=o.exclude_validset(xs); 
                  o.vxs=vxs;
                  o.txs=txs;
                  o.cxs=cxs;
              end
           end
            
        end
       
% monitoring       
       function [rbm,es,res]=cd1train(o)
            % no validsets
            if ~o.validsetq 
                if o.etadecq
                    [rbm,es,res]=o.rbm.cd1_etadecay(o.txs,o.monitori,o.eta0,o.etap,o.etam);
                elseif o.iterq          
                    [rbm,es,res]=o.rbm.cd1_etadecay(o.txs,o.monitori,o.eta0,0,o.eta0); % bit ugly
                else
                    [rbm,es,res]=o.rbm.cd1_while_energy_improve(o.txs,o.monitori);
                end
            else
                % music of the future - maybe YOU could implement it! :)
            end
        end
    end
end