# Moving-Obstacles-Search-Problem

The initial state of the problem is shown on the first image below. The player should move safely to to the house on the map. The player can move in every direction (up, down, left, right, upper-left, upper-right, lower-left, lower-right) as long player's position is in the borders of the map.

There are 3 moving obstacles (they make single move for every player's move):
  1. The first obstacle is moving horizontally and it's first move is on the left
  2. The second obstacle is moving diagonally and it's first move is on the upper-right
  3. The third obstacle is moving vertically and it's first move is down 
(example of the movements is shown on the second image below)

When an obstacle reaches the end of the map (the edge of the table), it should change it's direction and start moving again.The player is destroyed when colliding with an obstacle (when the player is on the same field with an obstacle).

The solution should work for any valid given position of the player. The user specifies the players coordinates after calling the script. Moreover, the solution should be based on an informed search algorithm that will find the lowest number of movements necessary for the player to reach the house.

![state1](https://user-images.githubusercontent.com/15221488/34907727-34e7f4e8-f883-11e7-8f0d-710646df7e01.png)

![state2](https://user-images.githubusercontent.com/15221488/34907730-3cb14ad0-f883-11e7-9262-ecce862ac42d.png)
