# Проверка датасета

## Схема
  пять обязательных колонок идут первыми: ок
  строк: 1333

## Значения
  SI сходится с CC50 / IC50 во всех 1299 строках с обеими величинами

## DOI
  строк с DOI: 1333 из 1333
  формат DOI: все проходят

## SMILES
  все 1333 SMILES разбираются RDKit

## Дубликаты
  молекул более чем с одной записью: 35
  из них расходятся больше чем в 10 раз: 10

## Доступность источников
  ChEMBL REST API -> https://www.ebi.ac.uk/chembl/api/data/status.json
     доступен, код 200
  PubChem PUG-REST -> https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/MolecularFormula/JSON
     доступен, код 200
  NCBI E-utilities -> https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pcassay&retmode=json
     доступен, код 200

## Сверка с живым API
  перезапрошено строк: 20, значение совпало: 20

## Итог
  ошибок: 0, предупреждений: 0
