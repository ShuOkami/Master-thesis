mdp

module soft_task1

	srct1_1: [0..3] init 2;
	srct1_2: [-1..3] init 3;
	sd1: [0..3] init 3;
	sp1_1: [0..6] init 5;
	sp1_2: [-1..6] init 6;
	f1: [0..1] init 0;

	[soft1] srct1_1=2 & srct1_2=3 & sd1=3 & sp1_1=5 & sp1_2=6 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=2) & (sp1_1'=4) & (sp1_2'=5);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=3 & sp1_1=5 & sp1_2=6 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=2) & (sp1_1'=4) & (sp1_2'=5);

	[none] srct1_1=2 & srct1_2=3 & sd1=3 & sp1_1=5 & sp1_2=6 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=2) & (sp1_1'=4) & (sp1_2'=5);

	[soft1] srct1_1=1 & srct1_2=2 & sd1=2 & sp1_1=4 & sp1_2=5 -> 7/10 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=3) & (sp1_2'=4) + 3/10 : (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=3) & (sp1_2'=4);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=2 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=1) & (sp1_1'=3) & (sp1_2'=4);

	[none] srct1_1=1 & srct1_2=2 & sd1=2 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=1) & (sp1_1'=3) & (sp1_2'=4);

	[soft1] srct1_1=2 & srct1_2=3 & sd1=2 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=1) & (sp1_1'=3) & (sp1_2'=4);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=2 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=1) & (sp1_1'=3) & (sp1_2'=4);

	[none] srct1_1=2 & srct1_2=3 & sd1=2 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=1) & (sp1_1'=3) & (sp1_2'=4);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[none] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft1] srct1_1=1 & srct1_2=-1 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft2] srct1_1=1 & srct1_2=-1 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[none] srct1_1=1 & srct1_2=-1 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft1] srct1_1=1 & srct1_2=2 & sd1=1 & sp1_1=3 & sp1_2=4 -> 7/10 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3) + 3/10 : (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[none] srct1_1=1 & srct1_2=2 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft1] srct1_1=2 & srct1_2=3 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[none] srct1_1=2 & srct1_2=3 & sd1=1 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[soft2] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[none] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[none] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[none] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 9/10 : (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6) + 1/10 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 9/10 : (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6) + 1/10 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[soft2] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 9/10 : (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6) + 1/10 : (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[none] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 9/10 : (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6) + 1/10 : (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=1 & sp1_2=2 -> 9/10 : (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6) + 1/10 : (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[none] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=1 & sp1_2=2 -> 9/10 : (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6) + 1/10 : (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=1 & sp1_2=2 -> 9/10 : (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6) + 1/10 : (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[none] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=1 & sp1_2=2 -> 9/10 : (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6) + 1/10 : (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6);

	[soft2] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6);

	[none] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6);

	[none] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6);

	[none] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5) & (sp1_2'=6);


endmodule

module soft_task2

	srct2_1: [0..2] init 2;
	sd2: [0..2] init 2;
	sp2_1: [0..9] init 9;
	f2: [0..1] init 0;

	[soft2] srct2_1=2 & sd2=2 & sp2_1=9 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=8);

	[soft1] srct2_1=2 & sd2=2 & sp2_1=9 -> (f2'=0) & (srct2_1'=2) & (sd2'=1) & (sp2_1'=8);

	[none] srct2_1=2 & sd2=2 & sp2_1=9 -> (f2'=0) & (srct2_1'=2) & (sd2'=1) & (sp2_1'=8);

	[soft2] srct2_1=1 & sd2=1 & sp2_1=8 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=7);

	[soft1] srct2_1=1 & sd2=1 & sp2_1=8 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=7);

	[none] srct2_1=1 & sd2=1 & sp2_1=8 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=7);

	[soft2] srct2_1=2 & sd2=1 & sp2_1=8 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=7);

	[soft1] srct2_1=2 & sd2=1 & sp2_1=8 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=7);

	[none] srct2_1=2 & sd2=1 & sp2_1=8 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=7);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=7 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=6);

	[none] srct2_1=0 & sd2=0 & sp2_1=7 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=6);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=7 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=6);

	[none] srct2_1=1 & sd2=0 & sp2_1=7 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=6);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=7 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=6);

	[none] srct2_1=2 & sd2=0 & sp2_1=7 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=6);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=6 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=5);

	[none] srct2_1=0 & sd2=0 & sp2_1=6 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=5);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=6 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=5);

	[none] srct2_1=1 & sd2=0 & sp2_1=6 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=5);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=6 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=5);

	[none] srct2_1=2 & sd2=0 & sp2_1=6 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=5);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=5 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=4);

	[none] srct2_1=0 & sd2=0 & sp2_1=5 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=4);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=5 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=4);

	[none] srct2_1=1 & sd2=0 & sp2_1=5 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=4);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=5 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=4);

	[none] srct2_1=2 & sd2=0 & sp2_1=5 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=4);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=3);

	[none] srct2_1=0 & sd2=0 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=3);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=3);

	[none] srct2_1=1 & sd2=0 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=3);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=4 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=3);

	[none] srct2_1=2 & sd2=0 & sp2_1=4 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=3);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[none] srct2_1=0 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[none] srct2_1=1 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=2);

	[none] srct2_1=2 & sd2=0 & sp2_1=3 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=2);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=0 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=1 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=2 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=9);

	[none] srct2_1=0 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=9);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=9);

	[none] srct2_1=1 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=9);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=9);

	[none] srct2_1=2 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=9);


endmodule

rewards


	[soft1] f1= 1 : 3;
	[soft2] f1= 1 : 3;
	[none] f1= 1 : 3;

	[soft1] f2= 1 : 7;
	[soft2] f2= 1 : 7;
	[none] f2= 1 : 7;

endrewards
