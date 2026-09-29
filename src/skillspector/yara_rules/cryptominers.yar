/*
    Crypto miner detection rules for source code scanning.
    Based on patterns from Neo23x0/signature-base and community threat intelligence.
    Covers stratum protocol, known pools, mining software references, and browser-based miners.
*/

rule crypto_stratum_protocol
{
    meta:
        description = "Stratum mining protocol usage (stratum+tcp/ssl, mining.subscribe/authorize)"
        category = "cryptominer"
        severity = "HIGH"
        confidence = "0.9"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $stratum_tcp  = "stratum+tcp://" nocase
        $stratum_ssl  = "stratum+ssl://" nocase
        $mining_sub   = "mining.subscribe" nocase
        $mining_auth  = "mining.authorize" nocase
        $mining_submit = "mining.submit" nocase
    condition:
        any of them
}

rule crypto_mining_pools
{
    meta:
        description = "Connection to known cryptocurrency mining pools"
        category = "cryptominer"
        severity = "HIGH"
        confidence = "0.85"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $pool_minexmr    = "pool.minexmr.com" nocase
        $pool_xmrpool    = "xmrpool.eu" nocase
        $pool_monero     = "monerohash.com" nocase
        $pool_supportxmr = "supportxmr.com" nocase
        $pool_nanopool   = "nanopool.org" nocase
        $pool_hashvault  = "hashvault.pro" nocase
        $pool_2miners    = "2miners.com" nocase
        $pool_herominers = "herominers.com" nocase
        $pool_unmine     = "unmineable.com" nocase
        $pool_nicehash   = "nicehash.com" nocase
        $pool_minergate  = "minergate.com" nocase
        $pool_f2pool     = "f2pool.com" nocase
        $pool_antpool    = "antpool.com" nocase
        $pool_viabtc     = "viabtc.com" nocase
        $pool_ethermine  = "ethermine.org" nocase
        $pool_flexpool   = "flexpool.io" nocase
        $pool_hiveon     = "hiveon.net" nocase
        $pool_ezil       = "ezil.me" nocase
    condition:
        any of them
}

rule crypto_miner_software
{
    meta:
        description = "References to known cryptocurrency mining software"
        category = "cryptominer"
        severity = "HIGH"
        confidence = "0.8"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $xmrig        = "xmrig" nocase
        $xmr_stak     = "xmr-stak" nocase
        $cpuminer     = "cpuminer" nocase
        $cgminer      = "cgminer" nocase
        $bfgminer     = "bfgminer" nocase
        $ethminer     = "ethminer" nocase
        $nbminer      = "nbminer" nocase
        $phoenixminer = "phoenixminer" nocase
        $t_rex_miner  = "t-rex" nocase
        $cryptonight  = "cryptonight" nocase
        $randomx      = "randomx" nocase
    condition:
        2 of them
}

rule crypto_coinjacking
{
    meta:
        description = "Browser-based cryptojacking scripts (CoinHive, CryptoLoot, etc.)"
        category = "cryptominer"
        severity = "CRITICAL"
        confidence = "0.9"
        reference = "https://github.com/Neo23x0/signature-base"
    strings:
        $coinhive_js   = "coinhive.min.js" nocase
        $coinhive_anon = /CoinHive\.Anonymous\s*\(/ nocase
        $cryptoloot    = "cryptoloot" nocase
        $webmine_pro   = "webmine.pro" nocase
        $jsecoin       = "jsecoin" nocase
        $coin_imp      = "coin-imp" nocase
        $minero_cc     = "minero.cc" nocase
        $monerominer   = "monerominer" nocase
        $wasm_miner    = /WebAssembly\.instantiate.{0,400}(\b(mine|miner)\b|mining\s*\(|cryptonight|randomx|cn_?hash|hash_?cn|hashrate)/ nocase
    condition:
        any of them
}
