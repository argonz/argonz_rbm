
rdo=rbmdbno();
rdo=rdo.add_layer(16,'Gzu',100,'BB',{'eta',0.01,'sprt',0.06,'pretrainc',200,'trainc',400});
rdo=rdo.add_layer_n(2,100,'BB',100,'BB',{'eta',0.05,'sprt',0.06,'pretrainc',250,'trainc',250});
rdo=rdo.cd1trains(xs2);
rdo.prob_repr(xs2(1:10,:))
