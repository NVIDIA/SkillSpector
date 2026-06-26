/*
   用于源代码扫描的加密挖矿检测规则。 
   基于 Neo23x0/signature-base 和社区威胁情报中的模式。 
   覆盖 Stratum 协议、已知矿池、挖矿软件引用以及基于浏览器的挖矿程序。
*/

rule crypto_stratum_protocol
{
    meta:
        description = "检测 Stratum 加密货币挖矿协议通信特征"
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
        description = "检测与已知加密货币矿池建立连接的行为"
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
        description = "检测加密货币挖矿程序及相关组件特征"
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
        description = "检测网页挖矿（Cryptojacking）脚本及相关恶意代码"
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
        $wasm_miner    = /WebAssembly\.instantiate.*(mine|hash|crypto)/
    condition:
        any of them
}
