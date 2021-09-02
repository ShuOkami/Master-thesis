mdp

module soft_task1

	srct1_1: [0..2] init 1;
	srct1_2: [-1..2] init 2;
	sd1: [0..5] init 5;
	sp1_1: [0..8] init 8;
	f1: [0..1] init 0;

	[soft1] srct1_1=1 & srct1_2=2 & sd1=5 & sp1_1=8 -> 3/5 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=4) & (sp1_1'=7) + 2/5 : (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=4) & (sp1_1'=7);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=4 & sp1_1=7 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=6);

	[none] srct1_1=0 & srct1_2=-1 & sd1=4 & sp1_1=7 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=6);

	[soft1] srct1_1=1 & srct1_2=-1 & sd1=4 & sp1_1=7 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=6);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=3 & sp1_1=6 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=2) & (sp1_1'=5);

	[none] srct1_1=0 & srct1_2=-1 & sd1=3 & sp1_1=6 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=2) & (sp1_1'=5);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=2 & sp1_1=5 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=4);

	[none] srct1_1=0 & srct1_2=-1 & sd1=2 & sp1_1=5 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=4);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=3);

	[none] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=3);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=8);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=8);


endmodule

module soft_task2

	srct2_1: [0..2] init 1;
	srct2_2: [-1..2] init 2;
	sd2: [0..7] init 7;
	sp2_1: [0..8] init 8;
	f2: [0..1] init 0;

	[soft2] srct2_1=1 & srct2_2=2 & sd2=7 & sp2_1=8 -> 1/2 : (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=6) & (sp2_1'=7) + 1/2 : (f2'=0) & (srct2_1'=1) & (srct2_2'=-1) & (sd2'=6) & (sp2_1'=7);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=6 & sp2_1=7 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=5) & (sp2_1'=6);

	[none] srct2_1=0 & srct2_2=-1 & sd2=6 & sp2_1=7 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=5) & (sp2_1'=6);

	[soft2] srct2_1=1 & srct2_2=-1 & sd2=6 & sp2_1=7 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=5) & (sp2_1'=6);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=5 & sp2_1=6 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=4) & (sp2_1'=5);

	[none] srct2_1=0 & srct2_2=-1 & sd2=5 & sp2_1=6 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=4) & (sp2_1'=5);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=4 & sp2_1=5 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=3) & (sp2_1'=4);

	[none] srct2_1=0 & srct2_2=-1 & sd2=4 & sp2_1=5 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=3) & (sp2_1'=4);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=3 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=2) & (sp2_1'=3);

	[none] srct2_1=0 & srct2_2=-1 & sd2=3 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=2) & (sp2_1'=3);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=2 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=1) & (sp2_1'=2);

	[none] srct2_1=0 & srct2_2=-1 & sd2=2 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=1) & (sp2_1'=2);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=1 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=0 & srct2_2=-1 & sd2=1 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=1) & (srct2_2'=2) & (sd2'=7) & (sp2_1'=8);

	[none] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=1) & (srct2_2'=2) & (sd2'=7) & (sp2_1'=8);


endmodule

rewards

	f1= 1 : 12;
	f2= 1 : 8;

endrewards
