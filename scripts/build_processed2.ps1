$ErrorActionPreference = "Stop"

$Python = "C:\ProgramData\anaconda3\python.exe"
$Datasets = @("BitcoinOTC-1", "BitcoinAlpha-1", "wikirfa", "epinions")

foreach ($Dataset in $Datasets) {
    Write-Host "== Building processed2 for $Dataset =="
    & $Python -c "from utils import get_data; data,tr,val,te=get_data('$Dataset','./data/$Dataset','cpu',processed_dir='processed2'); print('$Dataset', 'events=', data.num_events, 'nodes=', data.num_nodes, 'splits=', tr.num_events, val.num_events, te.num_events, 'pos_frac=', round(float(data.y.float().mean()), 4))"
}

