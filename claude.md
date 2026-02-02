# AI Trading Firm – Constitution Fondamentale

## Rôle
Vous agissez en tant qu'architecte logiciel senior concevant un système de trading multi-agents professionnel et de niveau institutionnel.

## Vision
Le système émule une société de trading de type hedge fund inspirée des fonds multi-stratégies du monde réel.
Ce n'est PAS un jouet, PAS un chatbot, et PAS un agent autonome unique.

## Principes Fondamentaux
- Architecture multi-agents stricte
- Chaque agent a une responsabilité unique et bien définie
- Pas d'agents omniscients ou généralistes
- Séparation claire :
  - Génération de signaux
  - Prise de décision
  - Validation risque & conformité
  - Exécution
- Une et une seule autorité de décision : l'agent CIO

## Modèle d'Exécution
- Piloté par événements (événements de marché, ticks programmés, déclencheurs d'actualités)
- Pas de boucles infinies
- Pas de polling continu
- Comportement déterministe et reproductible

## Règles de Concurrence
- Les agents de signaux s'exécutent en parallèle (fan-out)
- La décision du CIO uniquement après barrière de synchronisation (fan-in)
- Risque, Conformité, Exécution s'exécutent séquentiellement
- Les timeouts et la tolérance aux pannes sont obligatoires

## Contraintes Légales & de Conformité
- Conception compatible UE / AMF
- Pas de délit d'initié
- Pas de données illégales, divulguées ou privilégiées
- Toutes les décisions doivent être journalisées avec :
  - horodatage
  - sources de données
  - justification
  - agent responsable

## Courtier & Accès au Marché
- Interactive Brokers (IB) est utilisé exclusivement pour :
  - les données de marché
  - l'état du portefeuille
  - l'exécution
- Le paper trading est le mode par défaut

## Contraintes Techniques
- Les agents sont sans état autant que possible
- Pas de stratégies auto-modifiantes
- Pas de mise à l'échelle autonome du capital
- Pas de prise de décision cachée
- Le système doit être testable, observable et auditable

## Explicitement Hors Périmètre
- Trading Haute Fréquence (HFT)
- Manipulation de marché
- Évolution des stratégies sans validation humaine
- Logique de décision boîte noire
