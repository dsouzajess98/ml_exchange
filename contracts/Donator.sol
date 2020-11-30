pragma solidity ^0.4.11;

/// TUTORIAL CONTRACT DO NOT USE IN PRODUCTION
/// @title Donations collecting contract


contract Donator {

    uint public donationsTotal;
    uint public donationsUsd;
    uint public donationsCount;
    uint public defaultUsdRate;

    function Donator() {
        defaultUsdRate = 350;
    }

    // fallback function
    function () payable {
        donate(defaultUsdRate);
    }


    function donate(uint usd_rate) public payable {
        defaultUsdRate = usd_rate;
    }
}