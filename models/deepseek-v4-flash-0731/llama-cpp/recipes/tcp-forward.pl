#!/usr/bin/env perl
# Forward raw TCP bytes so SSE and ordinary llama-server HTTP share one delayed port.

use strict;
use warnings;
use IO::Select;
use IO::Socket::INET;

my ($listen_host, $listen_port, $target_host, $target_port) = @ARGV;
die "DeepSeek V4 TCP forwarder usage: tcp-forward.pl LISTEN_HOST LISTEN_PORT TARGET_HOST TARGET_PORT\n"
    unless defined $target_port;

my $listener = IO::Socket::INET->new(
    LocalAddr => $listen_host,
    LocalPort => $listen_port,
    Proto     => 'tcp',
    Listen    => 128,
    ReuseAddr => 1,
) or die "DeepSeek V4 TCP forwarder listen error on ${listen_host}:${listen_port}: $!\n";

$SIG{CHLD} = 'IGNORE';
$SIG{TERM} = sub { exit 0 };
$SIG{INT}  = sub { exit 0 };

sub write_all {
    my ($socket, $buffer) = @_;
    while (length $buffer) {
        my $written = syswrite($socket, $buffer);
        return 0 unless defined $written && $written > 0;
        substr($buffer, 0, $written, q{});
    }
    return 1;
}

while (my $client = $listener->accept()) {
    my $pid = fork();
    if (!defined $pid) {
        close $client;
        next;
    }
    if ($pid != 0) {
        close $client;
        next;
    }

    close $listener;
    my $target = IO::Socket::INET->new(
        PeerAddr => $target_host,
        PeerPort => $target_port,
        Proto     => 'tcp',
    );
    exit 1 unless $target;

    my $select = IO::Select->new($client, $target);
    while (my @readable = $select->can_read()) {
        for my $source (@readable) {
            my $buffer = q{};
            my $read = sysread($source, $buffer, 65_536);
            exit 0 unless defined $read && $read > 0;
            my $destination = fileno($source) == fileno($client) ? $target : $client;
            exit 0 unless write_all($destination, $buffer);
        }
    }
    exit 0;
}
