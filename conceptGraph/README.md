Main.py is the main function of the conceptGraph module. 

You can run this module by simply execute(but it requires external Stanford parser library for dependency parsing)
* python Main.py
The parameter of "./test/annotation_test.txt" is the input file path of raw knowledge containing sentences.
Command "m.output('./Evaluation/tmp.txt')" outputs the produced predicates & variables to files for evaluation with recall and precision

Main.py will output potential literals to file bi.test.literal and uni.test.literal in the same folder. 

Learning.py will learn a classifer from annotated data and make prediction for test data. The parameters setting can be found in comments. Accuracy, Prcision and F-measures are reported. 

##The format of literal file:
For example, here is a example in file "uni.test.literal"

An angle that equals to 90° is a right angle.
RightAngle-10(angle1-2)	0:2 1:8.0 2:-1 

The first line is the raw sentences and the second line is a potential valid literal.
RightAngle is a predicate with one parameter "angle1-2", where angle1 is a variable and -2 indicates the second term(angle) in raw sentences produces that variable. "0:2 1:8.0 2:-1 " is the list of features for classification. 

## The format of labled(annotated) file:
Adding the human annotation for each literal as:

If this literal is true:
An angle that equals to 90° is a right angle.
1	RightAngle-10(angle1-2)	0:2 1:8.0 2:-1 

If this literal is considerd invalid, those two lines remain the same. 

## Format of "console.output"
In case you cannot run our codes, we save the outputs to this file. 

an angle that equals to 90  > is a right angle.
angle is angle1	1 variable	 =>  If
angle is IsAngle	3 variable	 =>  If
angle is AngleOf	3 variable	 =>  If
equals is Equals	3 variable	 =>  If
90 is constant	0 variable	 =>  If
90 is Equals	3 variable	 =>  If
right is RightAngle	3 variable	 =>  Then
right is RightTriangle	3 variable	 =>  Then
angle is angle2	1 variable	 =>  Then
angle is IsAngle	3 variable	 =>  Then
angle is AngleOf	3 variable	 =>  Then
=== end of this sentence ====
An angle that equals to 90° is a right angle.
IsAngle( constant-6)
[3, 4.0, 3]
An angle that equals to 90° is a right angle.
IsAngle( angle1-2)
[0, 0.0, 0]
....
....
....

Here symbol ">"  indicate a if-then
The terms before ">" are from condition part and the terms behind that symbol are considered to be conclusions.
The prdicates and variable generatation process are shown in the following lines.
After line "=== end of this sentence ====" are potentail literal combinations which are similar to file "uni.test.literal"
