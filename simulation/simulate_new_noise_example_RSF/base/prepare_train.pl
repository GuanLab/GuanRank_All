#!/usr/bin/perl

open FILE, "new_feature.dat" or die;
while ($line=<FILE>){
    chomp $line;
    @table=split "\t", $line;
    $ref{$table[0]}=$line;
}
close FILE;

open TRAIN_GS, "gs_train.dat" or die;
open NEW, ">train.dat" or die;
while ($line=<TRAIN_GS>){
    chomp $line;
    @table=split "\t", $line;
    print NEW "$table[1]\t$table[2]";
    print "$table[1]\t$table[2]\n";
    @ttt=split "\t", $ref{$table[0]};
    shift @ttt;
    foreach $value (@ttt){
        print NEW "\t$value";
    }
    print NEW "\n";
}

close NEW;
close TRAIN_GS;

open TRAIN_GS, "gs_test.dat" or die;
open NEW, ">test.dat" or die;
while ($line=<TRAIN_GS>){
    chomp $line;
    @table=split "\t", $line;
    print NEW "$table[1]\t$table[2]";
    @ttt=split "\t", $ref{$table[0]};
    shift @ttt;
    foreach $value (@ttt){
        print NEW "\t$value";
    }
    print NEW "\n";
}

close NEW;
close TRAIN_GS;
