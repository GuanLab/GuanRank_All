#!/usr/bin/perl
#
%date=();
%status=();
open FILE, "gs_train.dat" or die;
while ($line=<FILE>){
    chomp $line;
    @table=split "\t", $line;
    $date{$table[0]}=$table[1];
    $status{$table[0]}=$table[2];
}
close FILE;


open CURVE, ">death_curve.txt" or die;

@dates=sort{$date{$a}<=>$date{$b}}keys%date;
$old_total=scalar(@dates);
$old_pid=shift @dates;
$old_date=$date{$old_pid};
$total=$old_total;
$original_live_ratio=1;
if ($status{$old_pid}==1){
    $death_count++;
}
$total--;
print CURVE "0\t1\n";
$curve{$old_date}=$original_live_ratio;
while (@dates){
    $pid=shift @dates;
    $new_date=$date{$pid};
    if ($new_date==$old_date){
        if ($status{$pid}==1){
            $death_count++;
        }
        $total--;
    }else{
        $current_live_ratio=1-$death_count/($total+$death_count);
        $original_live_ratio=$current_live_ratio*$original_live_ratio;
        print CURVE "$old_date\t$original_live_ratio\n";
        $curve{$old_date}=$original_live_ratio;
        $death_count=0;
        $old_total=$total;
        $old_date=$new_date;
        #if ($death{$pid}==1){
        if ($status{$pid}==1){
            $death_count++;
        }
        $total--;
    }
}
$curent_live_ratio=1-$death_count/($old_total+$death_count);
$original_live_ratio=$curent_live_ratio*$original_live_ratio;
$curve{$old_date}=$original_live_ratio;
print CURVE "$old_date\t$original_live_ratio\n";
close CURVE;



%rank=();
@pid=keys %date;
@pidcp=@pid;
foreach $p1 (@pid){
    foreach $p2 (@pidcp){
        if ($p1 lt $p2){
            if ($date{$p1}>$date{$p2}){
                if ($status{$p2}==1){
                    $rank{$p2}++;
                }else{
                    if ($status{$p1}==1){
                        $p=($curve{$date{$p2}}-$curve{$date{$p1}})/$curve{$date{$p2}};
                        $rank{$p2}+=$p;
                        $rank{$p1}+=(1-$p);
                    #	????
                    }else{
                        $p=($curve{$date{$p2}}-$curve{$date{$p1}})/$curve{$date{$p2}};
                        $rank{$p2}+=$p+0.5*(1-$p);
                        $rank{$p1}+=0.5*(1-$p);
                    }
                }
            }
            if ($date{$p1}<$date{$p2}){
                if ($status{$p1}==1){
                    $rank{$p1}++;
                }else{
                    if ($status{$p2}==1){
                        $p=($curve{$date{$p1}}-$curve{$date{$p2}})/$curve{$date{$p1}};
                        $rank{$p2}+=(1-$p);
                        $rank{$p1}+=$p;
						#	????
                    }else{
                        $p=($curve{$date{$p1}}-$curve{$date{$p2}})/$curve{$date{$p1}};
                        $rank{$p1}+=$p+0.5*(1-$p);
                        $rank{$p2}+=0.5*(1-$p);
                    }
                }
            }
            if ($date{$p1}==$date{$p2}){
                if (($status{$p1}==1)&&($status{$p2}==1)){
                    $rank{$p1}+=0.5;
                    $rank{$p2}+=0.5;
                }
                if (($status{$p1}==0)&&($status{$p2}==0)){
                    $rank{$p1}+=0.5;
                    $rank{$p2}+=0.5;
                }
                if (($status{$p1}==1)&&($status{$p2}==0)){
                    $rank{$p1}+=1;
                }
                if (($status{$p1}==0)&&($status{$p2}==1)){
                    $rank{$p2}+=1;
                }
            }
        }
    }
}

@all=sort{$rank{$b}<=>$rank{$a}}keys %rank;
            
open NEW, ">train_target.txt";
foreach $p (@pid){
    $val=$rank{$p}/$rank{$all[0]};
    print NEW "$p\t$val\n";
}
close NEW;

