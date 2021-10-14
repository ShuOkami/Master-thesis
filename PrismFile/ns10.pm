mdp

module soft_task1

	srct1_1: [0..1] init 1;
	sd1: [0..3] init 3;
	sp1_1: [0..5] init 5;
	f1: [0..1] init 0;

	[soft1] srct1_1=1 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=0) & (sd1'=2) & (sp1_1'=4);

	[soft2] srct1_1=1 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=1) & (sd1'=2) & (sp1_1'=4);

	[soft3] srct1_1=1 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=1) & (sd1'=2) & (sp1_1'=4);

	[soft4] srct1_1=1 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=1) & (sd1'=2) & (sp1_1'=4);

	[none] srct1_1=1 & sd1=3 & sp1_1=5 -> (f1'=0) & (srct1_1'=1) & (sd1'=2) & (sp1_1'=4);

	[soft2] srct1_1=0 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=0) & (sd1'=1) & (sp1_1'=3);

	[soft3] srct1_1=0 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=0) & (sd1'=1) & (sp1_1'=3);

	[soft4] srct1_1=0 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=0) & (sd1'=1) & (sp1_1'=3);

	[none] srct1_1=0 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=0) & (sd1'=1) & (sp1_1'=3);

	[soft1] srct1_1=1 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=0) & (sd1'=1) & (sp1_1'=3);

	[soft2] srct1_1=1 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=1) & (sd1'=1) & (sp1_1'=3);

	[soft3] srct1_1=1 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=1) & (sd1'=1) & (sp1_1'=3);

	[soft4] srct1_1=1 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=1) & (sd1'=1) & (sp1_1'=3);

	[none] srct1_1=1 & sd1=2 & sp1_1=4 -> (f1'=0) & (srct1_1'=1) & (sd1'=1) & (sp1_1'=3);

	[soft2] srct1_1=0 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=2);

	[soft3] srct1_1=0 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=2);

	[soft4] srct1_1=0 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=2);

	[none] srct1_1=0 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=2);

	[soft1] srct1_1=1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=2);

	[soft2] srct1_1=1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=2);

	[soft3] srct1_1=1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=2);

	[soft4] srct1_1=1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=2);

	[none] srct1_1=1 & sd1=1 & sp1_1=3 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=2);

	[soft2] srct1_1=0 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=1);

	[soft3] srct1_1=0 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=1);

	[soft4] srct1_1=0 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=0 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=0) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=1);

	[soft3] srct1_1=1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=1);

	[soft4] srct1_1=1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=1);

	[none] srct1_1=1 & sd1=0 & sp1_1=2 -> (f1'=0) & (srct1_1'=1) & (sd1'=0) & (sp1_1'=1);

	[soft2] srct1_1=0 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=1) & (sd1'=3) & (sp1_1'=5);

	[soft3] srct1_1=0 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=1) & (sd1'=3) & (sp1_1'=5);

	[soft4] srct1_1=0 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=1) & (sd1'=3) & (sp1_1'=5);

	[none] srct1_1=0 & sd1=0 & sp1_1=1 -> (f1'=0) & (srct1_1'=1) & (sd1'=3) & (sp1_1'=5);

	[soft2] srct1_1=1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=1) & (sd1'=3) & (sp1_1'=5);

	[soft3] srct1_1=1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=1) & (sd1'=3) & (sp1_1'=5);

	[soft4] srct1_1=1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=1) & (sd1'=3) & (sp1_1'=5);

	[none] srct1_1=1 & sd1=0 & sp1_1=1 -> (f1'=1) & (srct1_1'=1) & (sd1'=3) & (sp1_1'=5);


endmodule

module soft_task2

	srct2_1: [0..2] init 2;
	sd2: [0..4] init 4;
	sp2_1: [0..6] init 6;
	f2: [0..1] init 0;

	[soft2] srct2_1=2 & sd2=4 & sp2_1=6 -> (f2'=0) & (srct2_1'=1) & (sd2'=3) & (sp2_1'=5);

	[soft1] srct2_1=2 & sd2=4 & sp2_1=6 -> (f2'=0) & (srct2_1'=2) & (sd2'=3) & (sp2_1'=5);

	[soft3] srct2_1=2 & sd2=4 & sp2_1=6 -> (f2'=0) & (srct2_1'=2) & (sd2'=3) & (sp2_1'=5);

	[soft4] srct2_1=2 & sd2=4 & sp2_1=6 -> (f2'=0) & (srct2_1'=2) & (sd2'=3) & (sp2_1'=5);

	[none] srct2_1=2 & sd2=4 & sp2_1=6 -> (f2'=0) & (srct2_1'=2) & (sd2'=3) & (sp2_1'=5);

	[soft2] srct2_1=1 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=0) & (sd2'=2) & (sp2_1'=4);

	[soft1] srct2_1=1 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=1) & (sd2'=2) & (sp2_1'=4);

	[soft3] srct2_1=1 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=1) & (sd2'=2) & (sp2_1'=4);

	[soft4] srct2_1=1 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=1) & (sd2'=2) & (sp2_1'=4);

	[none] srct2_1=1 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=1) & (sd2'=2) & (sp2_1'=4);

	[soft2] srct2_1=2 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=1) & (sd2'=2) & (sp2_1'=4);

	[soft1] srct2_1=2 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=4);

	[soft3] srct2_1=2 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=4);

	[soft4] srct2_1=2 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=4);

	[none] srct2_1=2 & sd2=3 & sp2_1=5 -> (f2'=0) & (srct2_1'=2) & (sd2'=2) & (sp2_1'=4);

	[soft1] srct2_1=0 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=3);

	[soft3] srct2_1=0 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=3);

	[soft4] srct2_1=0 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=3);

	[none] srct2_1=0 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=3);

	[soft2] srct2_1=1 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=0) & (sd2'=1) & (sp2_1'=3);

	[soft1] srct2_1=1 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=3);

	[soft3] srct2_1=1 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=3);

	[soft4] srct2_1=1 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=3);

	[none] srct2_1=1 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=3);

	[soft2] srct2_1=2 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=1) & (sd2'=1) & (sp2_1'=3);

	[soft1] srct2_1=2 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=2) & (sd2'=1) & (sp2_1'=3);

	[soft3] srct2_1=2 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=2) & (sd2'=1) & (sp2_1'=3);

	[soft4] srct2_1=2 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=2) & (sd2'=1) & (sp2_1'=3);

	[none] srct2_1=2 & sd2=2 & sp2_1=4 -> (f2'=0) & (srct2_1'=2) & (sd2'=1) & (sp2_1'=3);

	[soft1] srct2_1=0 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[soft3] srct2_1=0 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[soft4] srct2_1=0 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[none] srct2_1=0 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[soft2] srct2_1=1 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=2);

	[soft1] srct2_1=1 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[soft3] srct2_1=1 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[soft4] srct2_1=1 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[none] srct2_1=1 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[soft2] srct2_1=2 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=2);

	[soft1] srct2_1=2 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=2);

	[soft3] srct2_1=2 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=2);

	[soft4] srct2_1=2 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=2);

	[none] srct2_1=2 & sd2=1 & sp2_1=3 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=2);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=1);

	[soft3] srct2_1=0 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=1);

	[soft4] srct2_1=0 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=0 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=0) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=1);

	[soft3] srct2_1=1 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=1);

	[soft4] srct2_1=1 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=1 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=1) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=1);

	[soft3] srct2_1=2 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=1);

	[soft4] srct2_1=2 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=1);

	[none] srct2_1=2 & sd2=0 & sp2_1=2 -> (f2'=0) & (srct2_1'=2) & (sd2'=0) & (sp2_1'=1);

	[soft1] srct2_1=0 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[soft3] srct2_1=0 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[soft4] srct2_1=0 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[none] srct2_1=0 & sd2=0 & sp2_1=1 -> (f2'=0) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[soft1] srct2_1=1 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[soft3] srct2_1=1 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[soft4] srct2_1=1 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[none] srct2_1=1 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[soft1] srct2_1=2 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[soft3] srct2_1=2 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[soft4] srct2_1=2 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);

	[none] srct2_1=2 & sd2=0 & sp2_1=1 -> (f2'=1) & (srct2_1'=2) & (sd2'=4) & (sp2_1'=6);


endmodule

module soft_task3

	srct3_1: [0..2] init 2;
	sd3: [0..4] init 4;
	sp3_1: [0..5] init 5;
	f3: [0..1] init 0;

	[soft3] srct3_1=2 & sd3=4 & sp3_1=5 -> (f3'=0) & (srct3_1'=1) & (sd3'=3) & (sp3_1'=4);

	[soft1] srct3_1=2 & sd3=4 & sp3_1=5 -> (f3'=0) & (srct3_1'=2) & (sd3'=3) & (sp3_1'=4);

	[soft2] srct3_1=2 & sd3=4 & sp3_1=5 -> (f3'=0) & (srct3_1'=2) & (sd3'=3) & (sp3_1'=4);

	[soft4] srct3_1=2 & sd3=4 & sp3_1=5 -> (f3'=0) & (srct3_1'=2) & (sd3'=3) & (sp3_1'=4);

	[none] srct3_1=2 & sd3=4 & sp3_1=5 -> (f3'=0) & (srct3_1'=2) & (sd3'=3) & (sp3_1'=4);

	[soft3] srct3_1=1 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=0) & (sd3'=2) & (sp3_1'=3);

	[soft1] srct3_1=1 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=3);

	[soft2] srct3_1=1 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=3);

	[soft4] srct3_1=1 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=3);

	[none] srct3_1=1 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=3);

	[soft3] srct3_1=2 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=1) & (sd3'=2) & (sp3_1'=3);

	[soft1] srct3_1=2 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=2) & (sd3'=2) & (sp3_1'=3);

	[soft2] srct3_1=2 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=2) & (sd3'=2) & (sp3_1'=3);

	[soft4] srct3_1=2 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=2) & (sd3'=2) & (sp3_1'=3);

	[none] srct3_1=2 & sd3=3 & sp3_1=4 -> (f3'=0) & (srct3_1'=2) & (sd3'=2) & (sp3_1'=3);

	[soft1] srct3_1=0 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=1) & (sp3_1'=2);

	[soft2] srct3_1=0 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=1) & (sp3_1'=2);

	[soft4] srct3_1=0 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=1) & (sp3_1'=2);

	[none] srct3_1=0 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=1) & (sp3_1'=2);

	[soft3] srct3_1=1 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=0) & (sd3'=1) & (sp3_1'=2);

	[soft1] srct3_1=1 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=1) & (sp3_1'=2);

	[soft2] srct3_1=1 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=1) & (sp3_1'=2);

	[soft4] srct3_1=1 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=1) & (sp3_1'=2);

	[none] srct3_1=1 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=1) & (sp3_1'=2);

	[soft3] srct3_1=2 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=1) & (sd3'=1) & (sp3_1'=2);

	[soft1] srct3_1=2 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=2) & (sd3'=1) & (sp3_1'=2);

	[soft2] srct3_1=2 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=2) & (sd3'=1) & (sp3_1'=2);

	[soft4] srct3_1=2 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=2) & (sd3'=1) & (sp3_1'=2);

	[none] srct3_1=2 & sd3=2 & sp3_1=3 -> (f3'=0) & (srct3_1'=2) & (sd3'=1) & (sp3_1'=2);

	[soft1] srct3_1=0 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=0) & (sp3_1'=1);

	[soft2] srct3_1=0 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=0) & (sp3_1'=1);

	[soft4] srct3_1=0 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=0) & (sp3_1'=1);

	[none] srct3_1=0 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=0) & (sp3_1'=1);

	[soft3] srct3_1=1 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=0) & (sd3'=0) & (sp3_1'=1);

	[soft1] srct3_1=1 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=0) & (sp3_1'=1);

	[soft2] srct3_1=1 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=0) & (sp3_1'=1);

	[soft4] srct3_1=1 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=0) & (sp3_1'=1);

	[none] srct3_1=1 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=0) & (sp3_1'=1);

	[soft3] srct3_1=2 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=1) & (sd3'=0) & (sp3_1'=1);

	[soft1] srct3_1=2 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=2) & (sd3'=0) & (sp3_1'=1);

	[soft2] srct3_1=2 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=2) & (sd3'=0) & (sp3_1'=1);

	[soft4] srct3_1=2 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=2) & (sd3'=0) & (sp3_1'=1);

	[none] srct3_1=2 & sd3=1 & sp3_1=2 -> (f3'=0) & (srct3_1'=2) & (sd3'=0) & (sp3_1'=1);

	[soft1] srct3_1=0 & sd3=0 & sp3_1=1 -> (f3'=0) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[soft2] srct3_1=0 & sd3=0 & sp3_1=1 -> (f3'=0) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[soft4] srct3_1=0 & sd3=0 & sp3_1=1 -> (f3'=0) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[none] srct3_1=0 & sd3=0 & sp3_1=1 -> (f3'=0) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[soft1] srct3_1=1 & sd3=0 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[soft2] srct3_1=1 & sd3=0 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[soft4] srct3_1=1 & sd3=0 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[none] srct3_1=1 & sd3=0 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[soft1] srct3_1=2 & sd3=0 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[soft2] srct3_1=2 & sd3=0 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[soft4] srct3_1=2 & sd3=0 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);

	[none] srct3_1=2 & sd3=0 & sp3_1=1 -> (f3'=1) & (srct3_1'=2) & (sd3'=4) & (sp3_1'=5);


endmodule

module soft_task4

	srct4_1: [0..1] init 1;
	sd4: [0..3] init 3;
	sp4_1: [0..4] init 4;
	f4: [0..1] init 0;

	[soft4] srct4_1=1 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=0) & (sd4'=2) & (sp4_1'=3);

	[soft1] srct4_1=1 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=1) & (sd4'=2) & (sp4_1'=3);

	[soft2] srct4_1=1 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=1) & (sd4'=2) & (sp4_1'=3);

	[soft3] srct4_1=1 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=1) & (sd4'=2) & (sp4_1'=3);

	[none] srct4_1=1 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=1) & (sd4'=2) & (sp4_1'=3);

	[soft1] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[soft2] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[soft3] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[none] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[soft4] srct4_1=1 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[soft1] srct4_1=1 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=1) & (sd4'=1) & (sp4_1'=2);

	[soft2] srct4_1=1 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=1) & (sd4'=1) & (sp4_1'=2);

	[soft3] srct4_1=1 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=1) & (sd4'=1) & (sp4_1'=2);

	[none] srct4_1=1 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=1) & (sd4'=1) & (sp4_1'=2);

	[soft1] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[soft2] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[soft3] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[none] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[soft4] srct4_1=1 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[soft1] srct4_1=1 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=1) & (sd4'=0) & (sp4_1'=1);

	[soft2] srct4_1=1 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=1) & (sd4'=0) & (sp4_1'=1);

	[soft3] srct4_1=1 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=1) & (sd4'=0) & (sp4_1'=1);

	[none] srct4_1=1 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=1) & (sd4'=0) & (sp4_1'=1);

	[soft1] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=3) & (sp4_1'=4);

	[soft2] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=3) & (sp4_1'=4);

	[soft3] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=3) & (sp4_1'=4);

	[none] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=3) & (sp4_1'=4);

	[soft1] srct4_1=1 & sd4=0 & sp4_1=1 -> (f4'=1) & (srct4_1'=1) & (sd4'=3) & (sp4_1'=4);

	[soft2] srct4_1=1 & sd4=0 & sp4_1=1 -> (f4'=1) & (srct4_1'=1) & (sd4'=3) & (sp4_1'=4);

	[soft3] srct4_1=1 & sd4=0 & sp4_1=1 -> (f4'=1) & (srct4_1'=1) & (sd4'=3) & (sp4_1'=4);

	[none] srct4_1=1 & sd4=0 & sp4_1=1 -> (f4'=1) & (srct4_1'=1) & (sd4'=3) & (sp4_1'=4);


endmodule

rewards


	[soft1] f1= 1 : 3;
	[soft2] f1= 1 : 3;
	[soft3] f1= 1 : 3;
	[soft4] f1= 1 : 3;
	[none] f1= 1 : 3;

	[soft1] f2= 1 : 7;
	[soft2] f2= 1 : 7;
	[soft3] f2= 1 : 7;
	[soft4] f2= 1 : 7;
	[none] f2= 1 : 7;

	[soft1] f3= 1 : 6;
	[soft2] f3= 1 : 6;
	[soft3] f3= 1 : 6;
	[soft4] f3= 1 : 6;
	[none] f3= 1 : 6;

	[soft1] f4= 1 : 2;
	[soft2] f4= 1 : 2;
	[soft3] f4= 1 : 2;
	[soft4] f4= 1 : 2;
	[none] f4= 1 : 2;

endrewards
