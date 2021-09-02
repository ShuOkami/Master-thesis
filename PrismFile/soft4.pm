mdp

module soft_task1

	srct1_1: [0..3] init 2;
	srct1_2: [-1..3] init 3;
	sd1: [0..3] init 3;
	sp1_1: [0..5] init 5;
	f1: [0..1] init 0;

	[soft1] srct1_1=2 & srct1_2=3 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=2) & (sp1_1'=4);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=2) & (sp1_1'=4);

	[soft3] srct1_1=2 & srct1_2=3 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=2) & (sp1_1'=4);

	[none] srct1_1=2 & srct1_2=3 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=2) & (sp1_1'=4);

	[soft1] srct1_1=1 & srct1_2=2 & sd1=2 & sp1_1=4 -> 7/10 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=3) + 3/10 : (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=3);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=1) & (sp1_1'=3);

	[soft3] srct1_1=1 & srct1_2=2 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=1) & (sp1_1'=3);

	[none] srct1_1=1 & srct1_2=2 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=1) & (sp1_1'=3);

	[soft1] srct1_1=2 & srct1_2=3 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=1) & (sp1_1'=3);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=1) & (sp1_1'=3);

	[soft3] srct1_1=2 & srct1_2=3 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=1) & (sp1_1'=3);

	[none] srct1_1=2 & srct1_2=3 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=1) & (sp1_1'=3);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[none] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[soft1] srct1_1=1 & srct1_2=-1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[soft2] srct1_1=1 & srct1_2=-1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[soft3] srct1_1=1 & srct1_2=-1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[none] srct1_1=1 & srct1_2=-1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[soft1] srct1_1=1 & srct1_2=2 & sd1=1 & sp1_1=3 -> 7/10 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) + 3/10 : (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=2);

	[soft3] srct1_1=1 & srct1_2=2 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=2);

	[none] srct1_1=1 & srct1_2=2 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=2);

	[soft1] srct1_1=2 & srct1_2=3 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=2);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=2);

	[soft3] srct1_1=2 & srct1_2=3 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=2);

	[none] srct1_1=2 & srct1_2=3 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=2);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1);

	[soft3] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=1);

	[soft3] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=1);

	[soft3] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[soft2] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[soft3] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[none] srct1_1=1 & srct1_2=-1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[soft2] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[soft3] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[none] srct1_1=1 & srct1_2=2 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[soft2] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[soft3] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);

	[none] srct1_1=2 & srct1_2=3 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=2) & (srct1_2'=3) & (sd1'=3) & (sp1_1'=5);


endmodule

module soft_task2

	srct2_1: [0..1] init 1;
	sd2: [0..3] init 3;
	sp2_1: [0..3] init 3;
	f2: [0..1] init 0;

	[soft2] srct2_1=1 & sd2=3 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=2) & (sp2_1'=2);

	[soft1] srct2_1=1 & sd2=3 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=2) & (sp2_1'=2);

	[soft3] srct2_1=1 & sd2=3 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=2) & (sp2_1'=2);

	[none] srct2_1=1 & sd2=3 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=2) & (sp2_1'=2);

	[soft1] srct2_1=0 & sd2=2 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=1);

	[soft3] srct2_1=0 & sd2=2 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=1);

	[none] srct2_1=0 & sd2=2 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=1);

	[soft2] srct2_1=1 & sd2=2 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=1);

	[soft1] srct2_1=1 & sd2=2 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=1);

	[soft3] srct2_1=1 & sd2=2 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=1);

	[none] srct2_1=1 & sd2=2 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=1);

	[soft1] srct2_1=0 & sd2=1 & sp2_1=1 -> (f2'=0) & (srct2_1'=1) & (sd2'=3) & (sp2_1'=3);

	[soft3] srct2_1=0 & sd2=1 & sp2_1=1 -> (f2'=0) & (srct2_1'=1) & (sd2'=3) & (sp2_1'=3);

	[none] srct2_1=0 & sd2=1 & sp2_1=1 -> (f2'=0) & (srct2_1'=1) & (sd2'=3) & (sp2_1'=3);

	[soft2] srct2_1=1 & sd2=1 & sp2_1=1 -> (f2'=0) & (srct2_1'=1) & (sd2'=3) & (sp2_1'=3);

	[soft1] srct2_1=1 & sd2=1 & sp2_1=1 -> (f2'=1) & (srct2_1'=1) & (sd2'=3) & (sp2_1'=3);

	[soft3] srct2_1=1 & sd2=1 & sp2_1=1 -> (f2'=1) & (srct2_1'=1) & (sd2'=3) & (sp2_1'=3);

	[none] srct2_1=1 & sd2=1 & sp2_1=1 -> (f2'=1) & (srct2_1'=1) & (sd2'=3) & (sp2_1'=3);


endmodule

module soft_task3

	srct3_1: [0..2] init 2;
	sd3: [0..6] init 6;
	sp3_1: [0..5] init 5;
	f3: [0..1] init 0;

	[soft3] srct3_1=2 & sd3=6 & sp3_1=5 -> (f3'=0) & (srct3_1'=1) & (sd3'=5) & (sp3_1'=4);

	[soft1] srct3_1=2 & sd3=6 & sp3_1=5 -> (f3'=0) & (srct3_1'=2) & (sd3'=5) & (sp3_1'=4);

	[soft2] srct3_1=2 & sd3=6 & sp3_1=5 -> (f3'=0) & (srct3_1'=2) & (sd3'=5) & (sp3_1'=4);

	[none] srct3_1=2 & sd3=6 & sp3_1=5 -> (f3'=0) & (srct3_1'=2) & (sd3'=5) & (sp3_1'=4);

	[soft3] srct3_1=1 & sd3=5 & sp3_1=4 -> (f3'=0) & (srct3_1'=0) & (sd3'=4) & (sp3_1'=3);

	[soft1] srct3_1=1 & sd3=5 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=4) & (sp3_1'=3);

	[soft2] srct3_1=1 & sd3=5 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=4) & (sp3_1'=3);

	[none] srct3_1=1 & sd3=5 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=4) & (sp3_1'=3);

	[soft3] srct3_1=2 & sd3=5 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=4) & (sp3_1'=3);

	[soft1] srct3_1=2 & sd3=5 & sp3_1=4 -> (f3'=0) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=3);

	[soft2] srct3_1=2 & sd3=5 & sp3_1=4 -> (f3'=0) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=3);

	[none] srct3_1=2 & sd3=5 & sp3_1=4 -> (f3'=0) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=3);

	[soft1] srct3_1=0 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=3) & (sp3_1'=2);

	[soft2] srct3_1=0 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=3) & (sp3_1'=2);

	[none] srct3_1=0 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=3) & (sp3_1'=2);

	[soft3] srct3_1=1 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=3) & (sp3_1'=2);

	[soft1] srct3_1=1 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=3) & (sp3_1'=2);

	[soft2] srct3_1=1 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=3) & (sp3_1'=2);

	[none] srct3_1=1 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=3) & (sp3_1'=2);

	[soft3] srct3_1=2 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=3) & (sp3_1'=2);

	[soft1] srct3_1=2 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=2) & (sd3'=3) & (sp3_1'=2);

	[soft2] srct3_1=2 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=2) & (sd3'=3) & (sp3_1'=2);

	[none] srct3_1=2 & sd3=4 & sp3_1=3 -> (f3'=0) & (srct3_1'=2) & (sd3'=3) & (sp3_1'=2);

	[soft1] srct3_1=0 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=2) & (sp3_1'=1);

	[soft2] srct3_1=0 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=2) & (sp3_1'=1);

	[none] srct3_1=0 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=2) & (sp3_1'=1);

	[soft3] srct3_1=1 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=2) & (sp3_1'=1);

	[soft1] srct3_1=1 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=1);

	[soft2] srct3_1=1 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=1);

	[none] srct3_1=1 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=1);

	[soft3] srct3_1=2 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=1);

	[soft1] srct3_1=2 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=2) & (sd3'=2) & (sp3_1'=1);

	[soft2] srct3_1=2 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=2) & (sd3'=2) & (sp3_1'=1);

	[none] srct3_1=2 & sd3=3 & sp3_1=2 -> (f3'=0) & (srct3_1'=2) & (sd3'=2) & (sp3_1'=1);

	[soft1] srct3_1=0 & sd3=2 & sp3_1=1 -> (f3'=0) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[soft2] srct3_1=0 & sd3=2 & sp3_1=1 -> (f3'=0) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[none] srct3_1=0 & sd3=2 & sp3_1=1 -> (f3'=0) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[soft3] srct3_1=1 & sd3=2 & sp3_1=1 -> (f3'=0) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[soft1] srct3_1=1 & sd3=2 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[soft2] srct3_1=1 & sd3=2 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[none] srct3_1=1 & sd3=2 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[soft1] srct3_1=2 & sd3=2 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[soft2] srct3_1=2 & sd3=2 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);

	[none] srct3_1=2 & sd3=2 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=6) & (sp3_1'=5);


endmodule

rewards


	[soft1] f1= 1 : 3;
	[soft2] f1= 1 : 3;
	[soft3] f1= 1 : 3;
	[none] f1= 1 : 3;

	[soft1] f2= 1 : 3;
	[soft2] f2= 1 : 3;
	[soft3] f2= 1 : 3;
	[none] f2= 1 : 3;

	[soft1] f3= 1 : 7;
	[soft2] f3= 1 : 7;
	[soft3] f3= 1 : 7;
	[none] f3= 1 : 7;

endrewards
