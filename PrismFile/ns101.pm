mdp

module soft_task1

	srct1_1: [0..3] init 3;
	sd1: [0..3] init 3;
	sp1_1: [0..4] init 4;
	f1: [0..1] init 0;

	[soft1] srct1_1=3 & sd1=3 & sp1_1=4 -> (f1'=0) & (srct1_1'=2) & (sd1'=2) & (sp1_1'=3);

	[soft2] srct1_1=3 & sd1=3 & sp1_1=4 -> (f1'=0) & (srct1_1'=3) & (sd1'=2) & (sp1_1'=3);

	[none] srct1_1=3 & sd1=3 & sp1_1=4 -> (f1'=0) & (srct1_1'=3) & (sd1'=2) & (sp1_1'=3);

	[soft1] srct1_1=2 & sd1=2 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (sd1'=1) & (sp1_1'=2);

	[soft2] srct1_1=2 & sd1=2 & sp1_1=3 -> (f1'=0) & (srct1_1'=2) & (sd1'=1) & (sp1_1'=2);

	[none] srct1_1=2 & sd1=2 & sp1_1=3 -> (f1'=0) & (srct1_1'=2) & (sd1'=1) & (sp1_1'=2);

	[soft1] srct1_1=3 & sd1=2 & sp1_1=3 -> (f1'=0) & (srct1_1'=2) & (sd1'=1) & (sp1_1'=2);

	[soft2] srct1_1=3 & sd1=2 & sp1_1=3 -> (f1'=0) & (srct1_1'=3) & (sd1'=1) & (sp1_1'=2);

	[none] srct1_1=3 & sd1=2 & sp1_1=3 -> (f1'=0) & (srct1_1'=3) & (sd1'=1) & (sp1_1'=2);

	[soft1] srct1_1=1 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=1 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=1 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=1);

	[soft1] srct1_1=2 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=2 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=2) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=2 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=2) & (sd1'=0) & (sp1_1'=1);

	[soft1] srct1_1=3 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=2) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=3 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=3) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=3 & sd1=1 & sp1_1=2 -> (f1'=0) & (srct1_1'=3) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=0 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=3) & (sd1'=3) & (sp1_1'=4);

	[none] srct1_1=0 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=3) & (sd1'=3) & (sp1_1'=4);

	[soft2] srct1_1=1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=3) & (sd1'=3) & (sp1_1'=4);

	[none] srct1_1=1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=3) & (sd1'=3) & (sp1_1'=4);

	[soft2] srct1_1=2 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=3) & (sd1'=3) & (sp1_1'=4);

	[none] srct1_1=2 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=3) & (sd1'=3) & (sp1_1'=4);

	[soft2] srct1_1=3 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=3) & (sd1'=3) & (sp1_1'=4);

	[none] srct1_1=3 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=3) & (sd1'=3) & (sp1_1'=4);


endmodule

module soft_task2

	srct2_1: [0..1] init 1;
	sd2: [0..1] init 1;
	sp2_1: [0..4] init 4;
	f2: [0..1] init 0;

	[soft2] srct2_1=1 & sd2=1 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=3);

	[soft1] srct2_1=1 & sd2=1 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=3);

	[none] srct2_1=1 & sd2=1 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=3);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[none] srct2_1=0 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[none] srct2_1=1 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=0 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=1 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=4);

	[none] srct2_1=0 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=4);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=4);

	[none] srct2_1=1 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=4);


endmodule

rewards


	[soft1] f1= 1 : 12;
	[soft2] f1= 1 : 12;
	[none] f1= 1 : 12;

	[soft1] f2= 1 : 8;
	[soft2] f2= 1 : 8;
	[none] f2= 1 : 8;

endrewards
