mdp

module soft_task1

	srct1_1: [0..2] init 1;
	srct1_2: [-1..2] init 2;
	sd1: [0..5] init 5;
	sp1_1: [0..11] init 10;
	sp1_2: [-1..11] init 11;
	f1: [0..1] init 0;

	[soft1] srct1_1=1 & srct1_2=2 & sd1=5 & sp1_1=10 & sp1_2=11 -> 1/5 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=4) & (sp1_1'=9) & (sp1_2'=10) + 4/5 : (f1'=0) & (srct1_1'=1) & (srct1_2'=-1) & (sd1'=4) & (sp1_1'=9) & (sp1_2'=10);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=4 & sp1_1=9 & sp1_2=10 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=8) & (sp1_2'=9);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=4 & sp1_1=9 & sp1_2=10 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=8) & (sp1_2'=9);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=4 & sp1_1=9 & sp1_2=10 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=8) & (sp1_2'=9);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=4 & sp1_1=9 & sp1_2=10 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=8) & (sp1_2'=9);

	[none] srct1_1=0 & srct1_2=-1 & sd1=4 & sp1_1=9 & sp1_2=10 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=8) & (sp1_2'=9);

	[soft1] srct1_1=1 & srct1_2=-1 & sd1=4 & sp1_1=9 & sp1_2=10 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=3) & (sp1_1'=8) & (sp1_2'=9);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=3 & sp1_1=8 & sp1_2=9 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=2) & (sp1_1'=7) & (sp1_2'=8);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=3 & sp1_1=8 & sp1_2=9 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=2) & (sp1_1'=7) & (sp1_2'=8);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=3 & sp1_1=8 & sp1_2=9 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=2) & (sp1_1'=7) & (sp1_2'=8);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=3 & sp1_1=8 & sp1_2=9 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=2) & (sp1_1'=7) & (sp1_2'=8);

	[none] srct1_1=0 & srct1_2=-1 & sd1=3 & sp1_1=8 & sp1_2=9 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=2) & (sp1_1'=7) & (sp1_2'=8);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=2 & sp1_1=7 & sp1_2=8 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=6) & (sp1_2'=7);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=2 & sp1_1=7 & sp1_2=8 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=6) & (sp1_2'=7);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=2 & sp1_1=7 & sp1_2=8 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=6) & (sp1_2'=7);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=2 & sp1_1=7 & sp1_2=8 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=6) & (sp1_2'=7);

	[none] srct1_1=0 & srct1_2=-1 & sd1=2 & sp1_1=7 & sp1_2=8 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=1) & (sp1_1'=6) & (sp1_2'=7);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=6 & sp1_2=7 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=5) & (sp1_2'=6);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=6 & sp1_2=7 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=5) & (sp1_2'=6);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=6 & sp1_2=7 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=5) & (sp1_2'=6);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=6 & sp1_2=7 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=5) & (sp1_2'=6);

	[none] srct1_1=0 & srct1_2=-1 & sd1=1 & sp1_1=6 & sp1_2=7 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=5) & (sp1_2'=6);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=5 & sp1_2=6 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=4) & (sp1_2'=5);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=5 & sp1_2=6 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=4) & (sp1_2'=5);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=5 & sp1_2=6 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=4) & (sp1_2'=5);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=5 & sp1_2=6 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=4) & (sp1_2'=5);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=5 & sp1_2=6 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=4) & (sp1_2'=5);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=3) & (sp1_2'=4);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=3) & (sp1_2'=4);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=3) & (sp1_2'=4);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=3) & (sp1_2'=4);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=4 & sp1_2=5 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=3) & (sp1_2'=4);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=3 & sp1_2=4 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=2) & (sp1_2'=3);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=2 & sp1_2=3 -> (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=2);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 2/5 : (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11) + 3/5 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 2/5 : (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11) + 3/5 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 2/5 : (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11) + 3/5 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 2/5 : (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11) + 3/5 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=2 -> 2/5 : (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11) + 3/5 : (f1'=0) & (srct1_1'=0) & (srct1_2'=-1) & (sd1'=0) & (sp1_1'=1) & (sp1_2'=-1);

	[soft2] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11);

	[soft3] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11);

	[soft4] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11);

	[soft5] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11);

	[none] srct1_1=0 & srct1_2=-1 & sd1=0 & sp1_1=1 & sp1_2=-1 -> (f1'=0) & (srct1_1'=1) & (srct1_2'=2) & (sd1'=5) & (sp1_1'=10) & (sp1_2'=11);


endmodule

module soft_task2

	srct2_1: [0..3] init 2;
	srct2_2: [-1..3] init 3;
	sd2: [0..7] init 7;
	sp2_1: [0..11] init 10;
	sp2_2: [-1..11] init 11;
	f2: [0..1] init 0;

	[soft2] srct2_1=2 & srct2_2=3 & sd2=7 & sp2_1=10 & sp2_2=11 -> (f2'=0) & (srct2_1'=1) & (srct2_2'=2) & (sd2'=6) & (sp2_1'=9) & (sp2_2'=10);

	[soft2] srct2_1=1 & srct2_2=2 & sd2=6 & sp2_1=9 & sp2_2=10 -> 4/5 : (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=5) & (sp2_1'=8) & (sp2_2'=9) + 1/5 : (f2'=0) & (srct2_1'=1) & (srct2_2'=-1) & (sd2'=5) & (sp2_1'=8) & (sp2_2'=9);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=5 & sp2_1=8 & sp2_2=9 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=4) & (sp2_1'=7) & (sp2_2'=8);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=5 & sp2_1=8 & sp2_2=9 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=4) & (sp2_1'=7) & (sp2_2'=8);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=5 & sp2_1=8 & sp2_2=9 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=4) & (sp2_1'=7) & (sp2_2'=8);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=5 & sp2_1=8 & sp2_2=9 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=4) & (sp2_1'=7) & (sp2_2'=8);

	[none] srct2_1=0 & srct2_2=-1 & sd2=5 & sp2_1=8 & sp2_2=9 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=4) & (sp2_1'=7) & (sp2_2'=8);

	[soft2] srct2_1=1 & srct2_2=-1 & sd2=5 & sp2_1=8 & sp2_2=9 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=4) & (sp2_1'=7) & (sp2_2'=8);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=4 & sp2_1=7 & sp2_2=8 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=3) & (sp2_1'=6) & (sp2_2'=7);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=4 & sp2_1=7 & sp2_2=8 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=3) & (sp2_1'=6) & (sp2_2'=7);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=4 & sp2_1=7 & sp2_2=8 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=3) & (sp2_1'=6) & (sp2_2'=7);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=4 & sp2_1=7 & sp2_2=8 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=3) & (sp2_1'=6) & (sp2_2'=7);

	[none] srct2_1=0 & srct2_2=-1 & sd2=4 & sp2_1=7 & sp2_2=8 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=3) & (sp2_1'=6) & (sp2_2'=7);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=3 & sp2_1=6 & sp2_2=7 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=2) & (sp2_1'=5) & (sp2_2'=6);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=3 & sp2_1=6 & sp2_2=7 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=2) & (sp2_1'=5) & (sp2_2'=6);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=3 & sp2_1=6 & sp2_2=7 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=2) & (sp2_1'=5) & (sp2_2'=6);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=3 & sp2_1=6 & sp2_2=7 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=2) & (sp2_1'=5) & (sp2_2'=6);

	[none] srct2_1=0 & srct2_2=-1 & sd2=3 & sp2_1=6 & sp2_2=7 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=2) & (sp2_1'=5) & (sp2_2'=6);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=2 & sp2_1=5 & sp2_2=6 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=1) & (sp2_1'=4) & (sp2_2'=5);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=2 & sp2_1=5 & sp2_2=6 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=1) & (sp2_1'=4) & (sp2_2'=5);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=2 & sp2_1=5 & sp2_2=6 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=1) & (sp2_1'=4) & (sp2_2'=5);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=2 & sp2_1=5 & sp2_2=6 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=1) & (sp2_1'=4) & (sp2_2'=5);

	[none] srct2_1=0 & srct2_2=-1 & sd2=2 & sp2_1=5 & sp2_2=6 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=1) & (sp2_1'=4) & (sp2_2'=5);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=1 & sp2_1=4 & sp2_2=5 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=3) & (sp2_2'=4);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=1 & sp2_1=4 & sp2_2=5 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=3) & (sp2_2'=4);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=1 & sp2_1=4 & sp2_2=5 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=3) & (sp2_2'=4);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=1 & sp2_1=4 & sp2_2=5 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=3) & (sp2_2'=4);

	[none] srct2_1=0 & srct2_2=-1 & sd2=1 & sp2_1=4 & sp2_2=5 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=3) & (sp2_2'=4);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=3 & sp2_2=4 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=2) & (sp2_2'=3);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=3 & sp2_2=4 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=2) & (sp2_2'=3);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=3 & sp2_2=4 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=2) & (sp2_2'=3);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=3 & sp2_2=4 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=2) & (sp2_2'=3);

	[none] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=3 & sp2_2=4 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=2) & (sp2_2'=3);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=2 & sp2_2=3 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=2);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=2 & sp2_2=3 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=2);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=2 & sp2_2=3 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=2);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=2 & sp2_2=3 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=2);

	[none] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=2 & sp2_2=3 -> (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=2);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=2 -> 1/2 : (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11) + 1/2 : (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=-1);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=2 -> 1/2 : (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11) + 1/2 : (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=-1);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=2 -> 1/2 : (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11) + 1/2 : (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=-1);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=2 -> 1/2 : (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11) + 1/2 : (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=-1);

	[none] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=2 -> 1/2 : (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11) + 1/2 : (f2'=0) & (srct2_1'=0) & (srct2_2'=-1) & (sd2'=0) & (sp2_1'=1) & (sp2_2'=-1);

	[soft1] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=-1 -> (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11);

	[soft3] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=-1 -> (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11);

	[soft4] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=-1 -> (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11);

	[soft5] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=-1 -> (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11);

	[none] srct2_1=0 & srct2_2=-1 & sd2=0 & sp2_1=1 & sp2_2=-1 -> (f2'=0) & (srct2_1'=2) & (srct2_2'=3) & (sd2'=7) & (sp2_1'=10) & (sp2_2'=11);


endmodule

module soft_task3

	srct3_1: [0..3] init 1;
	srct3_2: [-1..3] init 3;
	sd3: [0..8] init 8;
	sp3_1: [0..11] init 10;
	sp3_2: [-1..11] init 11;
	f3: [0..1] init 0;

	[soft3] srct3_1=1 & srct3_2=3 & sd3=8 & sp3_1=10 & sp3_2=11 -> 4/5 : (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=7) & (sp3_1'=9) & (sp3_2'=10) + 1/5 : (f3'=0) & (srct3_1'=2) & (srct3_2'=-1) & (sd3'=7) & (sp3_1'=9) & (sp3_2'=10);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=7 & sp3_1=9 & sp3_2=10 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=6) & (sp3_1'=8) & (sp3_2'=9);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=7 & sp3_1=9 & sp3_2=10 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=6) & (sp3_1'=8) & (sp3_2'=9);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=7 & sp3_1=9 & sp3_2=10 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=6) & (sp3_1'=8) & (sp3_2'=9);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=7 & sp3_1=9 & sp3_2=10 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=6) & (sp3_1'=8) & (sp3_2'=9);

	[none] srct3_1=0 & srct3_2=-1 & sd3=7 & sp3_1=9 & sp3_2=10 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=6) & (sp3_1'=8) & (sp3_2'=9);

	[soft3] srct3_1=2 & srct3_2=-1 & sd3=7 & sp3_1=9 & sp3_2=10 -> (f3'=0) & (srct3_1'=1) & (srct3_2'=-1) & (sd3'=6) & (sp3_1'=8) & (sp3_2'=9);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=6 & sp3_1=8 & sp3_2=9 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=5) & (sp3_1'=7) & (sp3_2'=8);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=6 & sp3_1=8 & sp3_2=9 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=5) & (sp3_1'=7) & (sp3_2'=8);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=6 & sp3_1=8 & sp3_2=9 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=5) & (sp3_1'=7) & (sp3_2'=8);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=6 & sp3_1=8 & sp3_2=9 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=5) & (sp3_1'=7) & (sp3_2'=8);

	[none] srct3_1=0 & srct3_2=-1 & sd3=6 & sp3_1=8 & sp3_2=9 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=5) & (sp3_1'=7) & (sp3_2'=8);

	[soft3] srct3_1=1 & srct3_2=-1 & sd3=6 & sp3_1=8 & sp3_2=9 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=5) & (sp3_1'=7) & (sp3_2'=8);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=5 & sp3_1=7 & sp3_2=8 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=4) & (sp3_1'=6) & (sp3_2'=7);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=5 & sp3_1=7 & sp3_2=8 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=4) & (sp3_1'=6) & (sp3_2'=7);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=5 & sp3_1=7 & sp3_2=8 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=4) & (sp3_1'=6) & (sp3_2'=7);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=5 & sp3_1=7 & sp3_2=8 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=4) & (sp3_1'=6) & (sp3_2'=7);

	[none] srct3_1=0 & srct3_2=-1 & sd3=5 & sp3_1=7 & sp3_2=8 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=4) & (sp3_1'=6) & (sp3_2'=7);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=4 & sp3_1=6 & sp3_2=7 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=3) & (sp3_1'=5) & (sp3_2'=6);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=4 & sp3_1=6 & sp3_2=7 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=3) & (sp3_1'=5) & (sp3_2'=6);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=4 & sp3_1=6 & sp3_2=7 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=3) & (sp3_1'=5) & (sp3_2'=6);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=4 & sp3_1=6 & sp3_2=7 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=3) & (sp3_1'=5) & (sp3_2'=6);

	[none] srct3_1=0 & srct3_2=-1 & sd3=4 & sp3_1=6 & sp3_2=7 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=3) & (sp3_1'=5) & (sp3_2'=6);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=3 & sp3_1=5 & sp3_2=6 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=2) & (sp3_1'=4) & (sp3_2'=5);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=3 & sp3_1=5 & sp3_2=6 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=2) & (sp3_1'=4) & (sp3_2'=5);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=3 & sp3_1=5 & sp3_2=6 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=2) & (sp3_1'=4) & (sp3_2'=5);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=3 & sp3_1=5 & sp3_2=6 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=2) & (sp3_1'=4) & (sp3_2'=5);

	[none] srct3_1=0 & srct3_2=-1 & sd3=3 & sp3_1=5 & sp3_2=6 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=2) & (sp3_1'=4) & (sp3_2'=5);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=2 & sp3_1=4 & sp3_2=5 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=1) & (sp3_1'=3) & (sp3_2'=4);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=2 & sp3_1=4 & sp3_2=5 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=1) & (sp3_1'=3) & (sp3_2'=4);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=2 & sp3_1=4 & sp3_2=5 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=1) & (sp3_1'=3) & (sp3_2'=4);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=2 & sp3_1=4 & sp3_2=5 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=1) & (sp3_1'=3) & (sp3_2'=4);

	[none] srct3_1=0 & srct3_2=-1 & sd3=2 & sp3_1=4 & sp3_2=5 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=1) & (sp3_1'=3) & (sp3_2'=4);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=1 & sp3_1=3 & sp3_2=4 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=2) & (sp3_2'=3);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=1 & sp3_1=3 & sp3_2=4 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=2) & (sp3_2'=3);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=1 & sp3_1=3 & sp3_2=4 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=2) & (sp3_2'=3);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=1 & sp3_1=3 & sp3_2=4 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=2) & (sp3_2'=3);

	[none] srct3_1=0 & srct3_2=-1 & sd3=1 & sp3_1=3 & sp3_2=4 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=2) & (sp3_2'=3);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=2 & sp3_2=3 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=2);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=2 & sp3_2=3 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=2);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=2 & sp3_2=3 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=2);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=2 & sp3_2=3 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=2);

	[none] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=2 & sp3_2=3 -> (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=2);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=2 -> 3/10 : (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11) + 7/10 : (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=-1);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=2 -> 3/10 : (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11) + 7/10 : (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=-1);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=2 -> 3/10 : (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11) + 7/10 : (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=-1);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=2 -> 3/10 : (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11) + 7/10 : (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=-1);

	[none] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=2 -> 3/10 : (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11) + 7/10 : (f3'=0) & (srct3_1'=0) & (srct3_2'=-1) & (sd3'=0) & (sp3_1'=1) & (sp3_2'=-1);

	[soft1] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=-1 -> (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11);

	[soft2] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=-1 -> (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11);

	[soft4] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=-1 -> (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11);

	[soft5] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=-1 -> (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11);

	[none] srct3_1=0 & srct3_2=-1 & sd3=0 & sp3_1=1 & sp3_2=-1 -> (f3'=0) & (srct3_1'=1) & (srct3_2'=3) & (sd3'=8) & (sp3_1'=10) & (sp3_2'=11);


endmodule

module soft_task4

	srct4_1: [0..1] init 1;
	sd4: [0..10] init 10;
	sp4_1: [0..11] init 11;
	f4: [0..1] init 0;

	[soft4] srct4_1=1 & sd4=10 & sp4_1=11 -> (f4'=0) & (srct4_1'=0) & (sd4'=9) & (sp4_1'=10);

	[soft1] srct4_1=0 & sd4=9 & sp4_1=10 -> (f4'=0) & (srct4_1'=0) & (sd4'=8) & (sp4_1'=9);

	[soft2] srct4_1=0 & sd4=9 & sp4_1=10 -> (f4'=0) & (srct4_1'=0) & (sd4'=8) & (sp4_1'=9);

	[soft3] srct4_1=0 & sd4=9 & sp4_1=10 -> (f4'=0) & (srct4_1'=0) & (sd4'=8) & (sp4_1'=9);

	[soft5] srct4_1=0 & sd4=9 & sp4_1=10 -> (f4'=0) & (srct4_1'=0) & (sd4'=8) & (sp4_1'=9);

	[none] srct4_1=0 & sd4=9 & sp4_1=10 -> (f4'=0) & (srct4_1'=0) & (sd4'=8) & (sp4_1'=9);

	[soft1] srct4_1=0 & sd4=8 & sp4_1=9 -> (f4'=0) & (srct4_1'=0) & (sd4'=7) & (sp4_1'=8);

	[soft2] srct4_1=0 & sd4=8 & sp4_1=9 -> (f4'=0) & (srct4_1'=0) & (sd4'=7) & (sp4_1'=8);

	[soft3] srct4_1=0 & sd4=8 & sp4_1=9 -> (f4'=0) & (srct4_1'=0) & (sd4'=7) & (sp4_1'=8);

	[soft5] srct4_1=0 & sd4=8 & sp4_1=9 -> (f4'=0) & (srct4_1'=0) & (sd4'=7) & (sp4_1'=8);

	[none] srct4_1=0 & sd4=8 & sp4_1=9 -> (f4'=0) & (srct4_1'=0) & (sd4'=7) & (sp4_1'=8);

	[soft1] srct4_1=0 & sd4=7 & sp4_1=8 -> (f4'=0) & (srct4_1'=0) & (sd4'=6) & (sp4_1'=7);

	[soft2] srct4_1=0 & sd4=7 & sp4_1=8 -> (f4'=0) & (srct4_1'=0) & (sd4'=6) & (sp4_1'=7);

	[soft3] srct4_1=0 & sd4=7 & sp4_1=8 -> (f4'=0) & (srct4_1'=0) & (sd4'=6) & (sp4_1'=7);

	[soft5] srct4_1=0 & sd4=7 & sp4_1=8 -> (f4'=0) & (srct4_1'=0) & (sd4'=6) & (sp4_1'=7);

	[none] srct4_1=0 & sd4=7 & sp4_1=8 -> (f4'=0) & (srct4_1'=0) & (sd4'=6) & (sp4_1'=7);

	[soft1] srct4_1=0 & sd4=6 & sp4_1=7 -> (f4'=0) & (srct4_1'=0) & (sd4'=5) & (sp4_1'=6);

	[soft2] srct4_1=0 & sd4=6 & sp4_1=7 -> (f4'=0) & (srct4_1'=0) & (sd4'=5) & (sp4_1'=6);

	[soft3] srct4_1=0 & sd4=6 & sp4_1=7 -> (f4'=0) & (srct4_1'=0) & (sd4'=5) & (sp4_1'=6);

	[soft5] srct4_1=0 & sd4=6 & sp4_1=7 -> (f4'=0) & (srct4_1'=0) & (sd4'=5) & (sp4_1'=6);

	[none] srct4_1=0 & sd4=6 & sp4_1=7 -> (f4'=0) & (srct4_1'=0) & (sd4'=5) & (sp4_1'=6);

	[soft1] srct4_1=0 & sd4=5 & sp4_1=6 -> (f4'=0) & (srct4_1'=0) & (sd4'=4) & (sp4_1'=5);

	[soft2] srct4_1=0 & sd4=5 & sp4_1=6 -> (f4'=0) & (srct4_1'=0) & (sd4'=4) & (sp4_1'=5);

	[soft3] srct4_1=0 & sd4=5 & sp4_1=6 -> (f4'=0) & (srct4_1'=0) & (sd4'=4) & (sp4_1'=5);

	[soft5] srct4_1=0 & sd4=5 & sp4_1=6 -> (f4'=0) & (srct4_1'=0) & (sd4'=4) & (sp4_1'=5);

	[none] srct4_1=0 & sd4=5 & sp4_1=6 -> (f4'=0) & (srct4_1'=0) & (sd4'=4) & (sp4_1'=5);

	[soft1] srct4_1=0 & sd4=4 & sp4_1=5 -> (f4'=0) & (srct4_1'=0) & (sd4'=3) & (sp4_1'=4);

	[soft2] srct4_1=0 & sd4=4 & sp4_1=5 -> (f4'=0) & (srct4_1'=0) & (sd4'=3) & (sp4_1'=4);

	[soft3] srct4_1=0 & sd4=4 & sp4_1=5 -> (f4'=0) & (srct4_1'=0) & (sd4'=3) & (sp4_1'=4);

	[soft5] srct4_1=0 & sd4=4 & sp4_1=5 -> (f4'=0) & (srct4_1'=0) & (sd4'=3) & (sp4_1'=4);

	[none] srct4_1=0 & sd4=4 & sp4_1=5 -> (f4'=0) & (srct4_1'=0) & (sd4'=3) & (sp4_1'=4);

	[soft1] srct4_1=0 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=0) & (sd4'=2) & (sp4_1'=3);

	[soft2] srct4_1=0 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=0) & (sd4'=2) & (sp4_1'=3);

	[soft3] srct4_1=0 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=0) & (sd4'=2) & (sp4_1'=3);

	[soft5] srct4_1=0 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=0) & (sd4'=2) & (sp4_1'=3);

	[none] srct4_1=0 & sd4=3 & sp4_1=4 -> (f4'=0) & (srct4_1'=0) & (sd4'=2) & (sp4_1'=3);

	[soft1] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[soft2] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[soft3] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[soft5] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[none] srct4_1=0 & sd4=2 & sp4_1=3 -> (f4'=0) & (srct4_1'=0) & (sd4'=1) & (sp4_1'=2);

	[soft1] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[soft2] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[soft3] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[soft5] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[none] srct4_1=0 & sd4=1 & sp4_1=2 -> (f4'=0) & (srct4_1'=0) & (sd4'=0) & (sp4_1'=1);

	[soft1] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=10) & (sp4_1'=11);

	[soft2] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=10) & (sp4_1'=11);

	[soft3] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=10) & (sp4_1'=11);

	[soft5] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=10) & (sp4_1'=11);

	[none] srct4_1=0 & sd4=0 & sp4_1=1 -> (f4'=0) & (srct4_1'=1) & (sd4'=10) & (sp4_1'=11);


endmodule

module soft_task5

	srct5_1: [0..2] init 2;
	sd5: [0..11] init 11;
	sp5_1: [0..11] init 11;
	f5: [0..1] init 0;

	[soft5] srct5_1=2 & sd5=11 & sp5_1=11 -> (f5'=0) & (srct5_1'=1) & (sd5'=10) & (sp5_1'=10);

	[soft5] srct5_1=1 & sd5=10 & sp5_1=10 -> (f5'=0) & (srct5_1'=0) & (sd5'=9) & (sp5_1'=9);

	[soft1] srct5_1=0 & sd5=9 & sp5_1=9 -> (f5'=0) & (srct5_1'=0) & (sd5'=8) & (sp5_1'=8);

	[soft2] srct5_1=0 & sd5=9 & sp5_1=9 -> (f5'=0) & (srct5_1'=0) & (sd5'=8) & (sp5_1'=8);

	[soft3] srct5_1=0 & sd5=9 & sp5_1=9 -> (f5'=0) & (srct5_1'=0) & (sd5'=8) & (sp5_1'=8);

	[soft4] srct5_1=0 & sd5=9 & sp5_1=9 -> (f5'=0) & (srct5_1'=0) & (sd5'=8) & (sp5_1'=8);

	[none] srct5_1=0 & sd5=9 & sp5_1=9 -> (f5'=0) & (srct5_1'=0) & (sd5'=8) & (sp5_1'=8);

	[soft1] srct5_1=0 & sd5=8 & sp5_1=8 -> (f5'=0) & (srct5_1'=0) & (sd5'=7) & (sp5_1'=7);

	[soft2] srct5_1=0 & sd5=8 & sp5_1=8 -> (f5'=0) & (srct5_1'=0) & (sd5'=7) & (sp5_1'=7);

	[soft3] srct5_1=0 & sd5=8 & sp5_1=8 -> (f5'=0) & (srct5_1'=0) & (sd5'=7) & (sp5_1'=7);

	[soft4] srct5_1=0 & sd5=8 & sp5_1=8 -> (f5'=0) & (srct5_1'=0) & (sd5'=7) & (sp5_1'=7);

	[none] srct5_1=0 & sd5=8 & sp5_1=8 -> (f5'=0) & (srct5_1'=0) & (sd5'=7) & (sp5_1'=7);

	[soft1] srct5_1=0 & sd5=7 & sp5_1=7 -> (f5'=0) & (srct5_1'=0) & (sd5'=6) & (sp5_1'=6);

	[soft2] srct5_1=0 & sd5=7 & sp5_1=7 -> (f5'=0) & (srct5_1'=0) & (sd5'=6) & (sp5_1'=6);

	[soft3] srct5_1=0 & sd5=7 & sp5_1=7 -> (f5'=0) & (srct5_1'=0) & (sd5'=6) & (sp5_1'=6);

	[soft4] srct5_1=0 & sd5=7 & sp5_1=7 -> (f5'=0) & (srct5_1'=0) & (sd5'=6) & (sp5_1'=6);

	[none] srct5_1=0 & sd5=7 & sp5_1=7 -> (f5'=0) & (srct5_1'=0) & (sd5'=6) & (sp5_1'=6);

	[soft1] srct5_1=0 & sd5=6 & sp5_1=6 -> (f5'=0) & (srct5_1'=0) & (sd5'=5) & (sp5_1'=5);

	[soft2] srct5_1=0 & sd5=6 & sp5_1=6 -> (f5'=0) & (srct5_1'=0) & (sd5'=5) & (sp5_1'=5);

	[soft3] srct5_1=0 & sd5=6 & sp5_1=6 -> (f5'=0) & (srct5_1'=0) & (sd5'=5) & (sp5_1'=5);

	[soft4] srct5_1=0 & sd5=6 & sp5_1=6 -> (f5'=0) & (srct5_1'=0) & (sd5'=5) & (sp5_1'=5);

	[none] srct5_1=0 & sd5=6 & sp5_1=6 -> (f5'=0) & (srct5_1'=0) & (sd5'=5) & (sp5_1'=5);

	[soft1] srct5_1=0 & sd5=5 & sp5_1=5 -> (f5'=0) & (srct5_1'=0) & (sd5'=4) & (sp5_1'=4);

	[soft2] srct5_1=0 & sd5=5 & sp5_1=5 -> (f5'=0) & (srct5_1'=0) & (sd5'=4) & (sp5_1'=4);

	[soft3] srct5_1=0 & sd5=5 & sp5_1=5 -> (f5'=0) & (srct5_1'=0) & (sd5'=4) & (sp5_1'=4);

	[soft4] srct5_1=0 & sd5=5 & sp5_1=5 -> (f5'=0) & (srct5_1'=0) & (sd5'=4) & (sp5_1'=4);

	[none] srct5_1=0 & sd5=5 & sp5_1=5 -> (f5'=0) & (srct5_1'=0) & (sd5'=4) & (sp5_1'=4);

	[soft1] srct5_1=0 & sd5=4 & sp5_1=4 -> (f5'=0) & (srct5_1'=0) & (sd5'=3) & (sp5_1'=3);

	[soft2] srct5_1=0 & sd5=4 & sp5_1=4 -> (f5'=0) & (srct5_1'=0) & (sd5'=3) & (sp5_1'=3);

	[soft3] srct5_1=0 & sd5=4 & sp5_1=4 -> (f5'=0) & (srct5_1'=0) & (sd5'=3) & (sp5_1'=3);

	[soft4] srct5_1=0 & sd5=4 & sp5_1=4 -> (f5'=0) & (srct5_1'=0) & (sd5'=3) & (sp5_1'=3);

	[none] srct5_1=0 & sd5=4 & sp5_1=4 -> (f5'=0) & (srct5_1'=0) & (sd5'=3) & (sp5_1'=3);

	[soft1] srct5_1=0 & sd5=3 & sp5_1=3 -> (f5'=0) & (srct5_1'=0) & (sd5'=2) & (sp5_1'=2);

	[soft2] srct5_1=0 & sd5=3 & sp5_1=3 -> (f5'=0) & (srct5_1'=0) & (sd5'=2) & (sp5_1'=2);

	[soft3] srct5_1=0 & sd5=3 & sp5_1=3 -> (f5'=0) & (srct5_1'=0) & (sd5'=2) & (sp5_1'=2);

	[soft4] srct5_1=0 & sd5=3 & sp5_1=3 -> (f5'=0) & (srct5_1'=0) & (sd5'=2) & (sp5_1'=2);

	[none] srct5_1=0 & sd5=3 & sp5_1=3 -> (f5'=0) & (srct5_1'=0) & (sd5'=2) & (sp5_1'=2);

	[soft1] srct5_1=0 & sd5=2 & sp5_1=2 -> (f5'=0) & (srct5_1'=0) & (sd5'=1) & (sp5_1'=1);

	[soft2] srct5_1=0 & sd5=2 & sp5_1=2 -> (f5'=0) & (srct5_1'=0) & (sd5'=1) & (sp5_1'=1);

	[soft3] srct5_1=0 & sd5=2 & sp5_1=2 -> (f5'=0) & (srct5_1'=0) & (sd5'=1) & (sp5_1'=1);

	[soft4] srct5_1=0 & sd5=2 & sp5_1=2 -> (f5'=0) & (srct5_1'=0) & (sd5'=1) & (sp5_1'=1);

	[none] srct5_1=0 & sd5=2 & sp5_1=2 -> (f5'=0) & (srct5_1'=0) & (sd5'=1) & (sp5_1'=1);

	[soft1] srct5_1=0 & sd5=1 & sp5_1=1 -> (f5'=0) & (srct5_1'=2) & (sd5'=11) & (sp5_1'=11);

	[soft2] srct5_1=0 & sd5=1 & sp5_1=1 -> (f5'=0) & (srct5_1'=2) & (sd5'=11) & (sp5_1'=11);

	[soft3] srct5_1=0 & sd5=1 & sp5_1=1 -> (f5'=0) & (srct5_1'=2) & (sd5'=11) & (sp5_1'=11);

	[soft4] srct5_1=0 & sd5=1 & sp5_1=1 -> (f5'=0) & (srct5_1'=2) & (sd5'=11) & (sp5_1'=11);

	[none] srct5_1=0 & sd5=1 & sp5_1=1 -> (f5'=0) & (srct5_1'=2) & (sd5'=11) & (sp5_1'=11);


endmodule

rewards


	[soft1] f1= 1 : 6;
	[soft2] f1= 1 : 6;
	[soft3] f1= 1 : 6;
	[soft4] f1= 1 : 6;
	[soft5] f1= 1 : 6;
	[none] f1= 1 : 6;

	[soft1] f2= 1 : 10;
	[soft2] f2= 1 : 10;
	[soft3] f2= 1 : 10;
	[soft4] f2= 1 : 10;
	[soft5] f2= 1 : 10;
	[none] f2= 1 : 10;

	[soft1] f3= 1 : 8;
	[soft2] f3= 1 : 8;
	[soft3] f3= 1 : 8;
	[soft4] f3= 1 : 8;
	[soft5] f3= 1 : 8;
	[none] f3= 1 : 8;

	[soft1] f4= 1 : 2;
	[soft2] f4= 1 : 2;
	[soft3] f4= 1 : 2;
	[soft4] f4= 1 : 2;
	[soft5] f4= 1 : 2;
	[none] f4= 1 : 2;

	[soft1] f5= 1 : 4;
	[soft2] f5= 1 : 4;
	[soft3] f5= 1 : 4;
	[soft4] f5= 1 : 4;
	[soft5] f5= 1 : 4;
	[none] f5= 1 : 4;

endrewards
