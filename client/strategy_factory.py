from typing import Callable, List, Tuple, Set
import math
from env import *
from utils import *
from strategy_utils import (
    fork_environment,
    get_attackable_targets,
    get_legal_moves,
    get_state_score,
    step_with_action,
)

# 控制 MCTS 是否输出调试日志，设为 False 可关闭所有 [MCTS] 输出
MCTS_VERBOSE: bool = False


def _allocate_init_positions(
    board: "Board",
    player_id: int,
    piece_cnt: int,
    preferred_order: List[Tuple[int, int]],
) -> List[Point]:
    """Pick distinct walkable cells on the player's side; prefer scan order then full fallback."""
    occupied: Set[Tuple[int, int]] = set()
    out: List[Point] = []
    bdr = board.boarder

    def cell_ok(x: int, y: int) -> bool:
        if (x, y) in occupied:
            return False
        if not board.is_within_bounds(Point(x, y)):
            return False
        if board.grid[x][y].state != 1:
            return False
        if player_id == 1:
            return y < bdr
        return y > bdr

    for _ in range(piece_cnt):
        pos = None
        for x, y in preferred_order:
            if cell_ok(x, y):
                pos = Point(x, y)
                break
        if pos is None:
            for y in range(board.height):
                for x in range(board.width):
                    if cell_ok(x, y):
                        pos = Point(x, y)
                        break
                if pos is not None:
                    break
        if pos is None:
            raise RuntimeError("init placement: no free cell in player's half")
        out.append(pos)
        occupied.add((pos.x, pos.y))
    return out


class StrategyFactory:
    """策略工厂类 - 提供不同的游戏策略"""
    
    @staticmethod
    def calculate_distance(p1: Point, p2: Point) -> float:
        """计算两点之间的距离"""
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    @staticmethod
    def _enemies(env: Environment, piece: "Piece") -> List["Piece"]:
        return [
            other for other in env.action_queue
            if other.is_alive and other.team != piece.team
        ]

    @staticmethod
    def _allies(env: Environment, piece: "Piece") -> List["Piece"]:
        return [
            other for other in env.action_queue
            if other.is_alive and other.team == piece.team
        ]

    @staticmethod
    def _attack_damage(attacker: "Piece", target: "Piece") -> float:
        if getattr(attacker, "weapon_type", 0) == 4:
            return 4.0
        raw_damage = attacker.physical_damage + attacker.strength
        return float(max(0, raw_damage - target.physical_resist))

    @staticmethod
    def _enemy_threat(piece: "Piece") -> float:
        return (
            piece.physical_damage
            + piece.strength * 0.7
            + piece.attack_range * 0.8
            + piece.spell_slots * 3.0
            + piece.action_points * 2.0
        )

    @staticmethod
    def _target_score(env: Environment, attacker: "Piece", target: "Piece", pos: Point = None) -> float:
        if pos is None:
            pos = attacker.position
        distance = StrategyFactory.calculate_distance(pos, target.position)
        damage = StrategyFactory._attack_damage(attacker, target)
        score = 0.0
        score += max(0, target.max_health - target.health) * 0.35
        score += (target.max_health - target.health) / max(1, target.max_health) * 14.0
        score += StrategyFactory._enemy_threat(target) * 0.45
        score += max(0, 16 - distance) * 1.2
        score += damage * 1.5
        if damage >= target.health:
            score += 100.0
        if distance <= attacker.attack_range:
            score += 18.0
        return score

    @staticmethod
    def _best_target(env: Environment, attacker: "Piece", pos: Point = None) -> "Piece":
        enemies = StrategyFactory._enemies(env, attacker)
        if not enemies:
            return None
        return max(enemies, key=lambda e: StrategyFactory._target_score(env, attacker, e, pos))

    @staticmethod
    def _enemy_pressure(env: Environment, team: int, point: Point) -> float:
        pressure = 0.0
        for enemy in env.action_queue:
            if not enemy.is_alive or enemy.team == team:
                continue
            distance = StrategyFactory.calculate_distance(point, enemy.position)
            if distance <= enemy.attack_range:
                pressure += max(8.0, enemy.physical_damage + enemy.strength * 0.4)
            elif distance <= enemy.attack_range + enemy.movement:
                pressure += 4.0
            if enemy.spell_slots > 0 and distance <= 7:
                pressure += 4.0
        return pressure

    @staticmethod
    def _support_score(env: Environment, current: "Piece", point: Point) -> float:
        score = 0.0
        for ally in StrategyFactory._allies(env, current):
            if ally is current:
                continue
            distance = StrategyFactory.calculate_distance(point, ally.position)
            if distance <= 5:
                score += 5.0
            elif distance <= 9:
                score += 2.0
            if distance <= 2:
                score -= 3.5
        return score

    @staticmethod
    def _move_score(env: Environment, current: "Piece", point: Point, target: "Piece") -> float:
        score = 0.0
        if target is not None:
            distance = StrategyFactory.calculate_distance(point, target.position)
            score += max(0, 18 - distance) * 1.8
            if distance <= current.attack_range:
                score += 35.0
            if StrategyFactory._attack_damage(current, target) >= target.health and distance <= current.attack_range:
                score += 80.0
        score += env.board.get_height(point) * 2.0
        score += StrategyFactory._support_score(env, current, point)
        score -= StrategyFactory._enemy_pressure(env, current.team, point) * 1.15
        return score

    @staticmethod
    def _build_attack(current: "Piece", target: "Piece") -> ActionSet:
        action = ActionSet()
        action.move = False
        action.attack = True
        action.attack_context = AttackContext()
        action.attack_context.attacker = current
        action.attack_context.target = target
        action.attack_context.attack_position = current.position
        action.spell = False
        return action

    @staticmethod
    def _add_attack(action: ActionSet, current: "Piece", target: "Piece") -> None:
        action.attack = True
        action.attack_context = AttackContext()
        action.attack_context.attacker = current
        action.attack_context.target = target
        action.attack_context.attack_position = current.position

    @staticmethod
    def _spell_by_id(env: Environment, current: "Piece", spell_id: int) -> "Spell":
        for spell in env.get_available_spells(current):
            if spell.id == spell_id:
                return spell
        return None

    @staticmethod
    def _build_spell(current: "Piece", spell: "Spell", target: "Piece", center: Point) -> ActionSet:
        action = ActionSet()
        action.move = False
        action.attack = False
        action.spell = True
        action.spell_context = SpellContext()
        action.spell_context.caster = current
        action.spell_context.spell = spell
        action.spell_context.target = target
        action.spell_context.target_area = Area(center.x, center.y, spell.area_radius)
        action.spell_context.target_type = TargetType.AREA if spell.is_area_effect else TargetType.SINGLE
        action.spell_context.is_area_effect = spell.is_area_effect
        action.spell_context.is_locking_spell = spell.is_locking_spell
        action.spell_context.spell_cost = spell.spell_cost
        return action

    @staticmethod
    def _try_kill_spell(env: Environment, current: "Piece") -> ActionSet:
        if current.action_points <= 0 or current.spell_slots <= 0:
            return None
        arrow = StrategyFactory._spell_by_id(env, current, 3)
        if arrow is None:
            return None
        best = None
        for enemy in StrategyFactory._enemies(env, current):
            distance = StrategyFactory.calculate_distance(current.position, enemy.position)
            if distance <= arrow.range and arrow.base_value >= enemy.health:
                if best is None or enemy.health < best.health:
                    best = enemy
        if best is None:
            return None
        return StrategyFactory._build_spell(current, arrow, best, best.position)

    @staticmethod
    def _try_heal(env: Environment, current: "Piece") -> ActionSet:
        if current.action_points <= 0 or current.spell_slots <= 0:
            return None
        heal = StrategyFactory._spell_by_id(env, current, 2)
        if heal is None:
            return None
        injured = [
            ally for ally in StrategyFactory._allies(env, current)
            if ally.health < ally.max_health - 8
            and StrategyFactory.calculate_distance(current.position, ally.position) <= heal.range
        ]
        if not injured:
            return None
        target = min(injured, key=lambda p: p.health / max(1, p.max_health))
        if target.health / max(1, target.max_health) > 0.55:
            return None
        return StrategyFactory._build_spell(current, heal, None, target.position)

    @staticmethod
    def _should_retreat(env: Environment, current: "Piece") -> bool:
        if current.health / max(1, current.max_health) <= 0.35:
            return True
        return StrategyFactory._enemy_pressure(env, current.team, current.position) >= current.health

    @staticmethod
    def get_custom_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            mid_x = board.width // 2
            offsets = [0, -3, 3, -5, 5, -1, 1, -7, 7]
            if pid == 1:
                order = [
                    (mid_x + dx, board.boarder - 1 - min(i, 2))
                    for i, dx in enumerate(offsets)
                ] + [
                    (x, y)
                    for y in range(board.boarder - 1, max(-1, board.boarder - 6), -1)
                    for x in sorted(range(1, board.width - 1), key=lambda v: abs(v - mid_x))
                ]
            else:
                order = [
                    (mid_x + dx, board.boarder + 1 + min(i, 2))
                    for i, dx in enumerate(offsets)
                ] + [
                    (x, y)
                    for y in range(board.boarder + 1, min(board.height, board.boarder + 6))
                    for x in sorted(range(1, board.width - 1), key=lambda v: abs(v - mid_x))
                ]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)

            builds = [
                (22, 4, 4, Point(1, 3)),
                (22, 4, 4, Point(2, 3)),
                (20, 6, 4, Point(3, 3)),
            ]
            piece_args: List[PieceArg] = []
            for i in range(init_message.piece_cnt):
                strength, dexterity, intelligence, equip = builds[min(i, len(builds) - 1)]
                arg = PieceArg()
                arg.strength = strength
                arg.dexterity = dexterity
                arg.intelligence = intelligence
                arg.equip = equip
                arg.pos = positions[i]
                piece_args.append(arg)
            return piece_args

        return strategy

    @staticmethod
    def get_custom_action_strategy() -> Callable[[Environment], ActionSet]:
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            action.move = False
            action.attack = False
            action.spell = False

            current = env.current_piece
            if current is None or not current.is_alive:
                return action

            enemies = StrategyFactory._enemies(env, current)
            if not enemies:
                return action

            healing_action = StrategyFactory._try_heal(env, current)
            if healing_action is not None:
                return healing_action

            attackable = [
                enemy for enemy in enemies
                if StrategyFactory.calculate_distance(current.position, enemy.position) <= current.attack_range
            ]
            kill_targets = [
                enemy for enemy in attackable
                if StrategyFactory._attack_damage(current, enemy) >= enemy.health
            ]
            if kill_targets:
                target = min(kill_targets, key=lambda p: p.health)
                return StrategyFactory._build_attack(current, target)

            spell_kill = StrategyFactory._try_kill_spell(env, current)
            if spell_kill is not None:
                return spell_kill

            focus = StrategyFactory._best_target(env, current)
            legal_moves = get_legal_moves(env)

            if StrategyFactory._should_retreat(env, current) and legal_moves:
                best_escape = max(
                    legal_moves,
                    key=lambda p: -StrategyFactory._enemy_pressure(env, current.team, p)
                    + StrategyFactory._support_score(env, current, p)
                    + env.board.get_height(p)
                )
                if StrategyFactory._enemy_pressure(env, current.team, best_escape) < StrategyFactory._enemy_pressure(env, current.team, current.position):
                    action.move = True
                    action.move_target = best_escape
                    return action

            if attackable:
                target = max(attackable, key=lambda p: StrategyFactory._target_score(env, current, p))
                return StrategyFactory._build_attack(current, target)

            best_move = None
            if legal_moves:
                limited_moves = sorted(
                    legal_moves,
                    key=lambda p: StrategyFactory._move_score(env, current, p, focus),
                    reverse=True,
                )[:10]
                best_move = max(
                    limited_moves,
                    key=lambda p: StrategyFactory._move_score(env, current, p, focus)
                )

            if best_move is not None:
                action.move = True
                action.move_target = best_move

                if current.max_action_points >= 2 and focus is not None:
                    after_move_targets = [
                        enemy for enemy in enemies
                        if StrategyFactory.calculate_distance(best_move, enemy.position) <= current.attack_range
                    ]
                    if after_move_targets:
                        target = max(after_move_targets, key=lambda p: StrategyFactory._target_score(env, current, p, best_move))
                        StrategyFactory._add_attack(action, current, target)
                return action

            return action

        return strategy

    @staticmethod
    def get_aggressive_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """获取攻击型初始化策略"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            if pid == 1:
                order = [
                    (x, y)
                    for y in range(5, 0, -1)
                    for x in range(2, board.width - 2)
                ]
            else:
                order = [
                    (x, y)
                    for y in range(board.height - 6, board.height)
                    for x in range(board.width - 3, 2, -1)
                ]
            positions = _allocate_init_positions(
                board, pid, init_message.piece_cnt, order
            )
            piece_args: List[PieceArg] = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 20
                arg.dexterity = 8
                arg.intelligence = 2
                arg.equip = Point(2, 3)
                arg.pos = pos
                piece_args.append(arg)
            return piece_args

        return strategy

    @staticmethod
    def get_defensive_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """获取防御型初始化策略"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            if pid == 1:
                order = [
                    (x, y)
                    for y in range(3, board.boarder)
                    for x in range(3, board.width - 3)
                ]
            else:
                order = [
                    (x, y)
                    for y in range(board.height - 1, board.boarder, -1)
                    for x in range(board.width - 4, 3, -1)
                ]
            positions = _allocate_init_positions(
                board, pid, init_message.piece_cnt, order
            )
            piece_args: List[PieceArg] = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 5
                arg.dexterity = 15
                arg.intelligence = 10
                arg.equip = Point(3, 1)
                arg.pos = pos
                piece_args.append(arg)
            return piece_args

        return strategy

    @staticmethod
    def get_aggressive_action_strategy() -> Callable[[Environment], ActionSet]:
        """获取攻击型行动策略 - 主动接近并攻击敌人"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            current_piece = env.current_piece
            
            # 寻找最近的敌人
            target_enemy = None
            nearest_distance = float('inf')
            
            for piece in env.action_queue:
                if piece.team != current_piece.team and piece.is_alive:
                    distance = StrategyFactory.calculate_distance(
                        Point(current_piece.position.x, current_piece.position.y),
                        Point(piece.position.x, piece.position.y)
                    )
                    if distance < nearest_distance:
                        nearest_distance = distance
                        target_enemy = piece
            
            # 没有敌人，不执行任何动作
            if target_enemy is None:
                action.move = False
                action.attack = False
                action.spell = False
                return action
            
            # 移动决策 - 向目标敌人移动
            # 获取所有合法移动位置
            legal_moves = get_legal_moves(env)
            if legal_moves:
                # 找到最接近敌人的合法移动位置
                best_move = None
                min_distance = float('inf')
                for move in legal_moves:
                    distance = StrategyFactory.calculate_distance(move, target_enemy.position)
                    if distance < min_distance:
                        min_distance = distance
                        best_move = move
                
                if best_move:
                    action.move = True
                    action.move_target = best_move
                else:
                    action.move = False
            else:
                action.move = False
            
            # 如果已经在攻击范围内，则攻击
            if nearest_distance <= current_piece.attack_range:
                action.attack = True
                action.attack_context = AttackContext()
                action.attack_context.attacker = current_piece
                action.attack_context.target = target_enemy
                action.attack_context.attackPosition = current_piece.position
            else:
                action.attack = False
            
            # 暂不使用法术
            action.spell = False
            
            return action
        
        return strategy

    @staticmethod
    def get_defensive_action_strategy() -> Callable[[Environment], ActionSet]:
        """获取防御型行动策略 - 保持距离，使用远程攻击"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            current_piece = env.current_piece
            
            # 寻找最近的敌人
            target_enemy = None
            nearest_distance = float('inf')
            
            for piece in env.action_queue:
                if piece.team != current_piece.team and piece.is_alive:
                    distance = StrategyFactory.calculate_distance(
                        Point(current_piece.position.x, current_piece.position.y),
                        Point(piece.position.x, piece.position.y)
                    )
                    if distance < nearest_distance:
                        nearest_distance = distance
                        target_enemy = piece
            
            # 没有敌人，不执行任何动作
            if target_enemy is None:
                action.move = False
                action.attack = False
                action.spell = False
                return action
            
            # 移动决策 - 保持在攻击范围内，但不要太近
            # 获取所有合法移动位置
            legal_moves = get_legal_moves(env)
            if legal_moves:
                # 理想距离是攻击范围的70%
                ideal_distance = current_piece.attack_range * 0.7
                
                # 找到最接近理想距离的合法移动位置
                best_move = None
                min_distance_diff = float('inf')
                
                for move in legal_moves:
                    distance_to_enemy = StrategyFactory.calculate_distance(move, target_enemy.position)
                    distance_diff = abs(distance_to_enemy - ideal_distance)
                    
                    if distance_diff < min_distance_diff:
                        # 如果太近了，只考虑能让我们远离的移动
                        if nearest_distance < ideal_distance - 2:
                            if distance_to_enemy > nearest_distance:
                                min_distance_diff = distance_diff
                                best_move = move
                        # 如果太远了，只考虑能让我们靠近的移动
                        elif nearest_distance > ideal_distance + 2:
                            if distance_to_enemy < nearest_distance:
                                min_distance_diff = distance_diff
                                best_move = move
                        # 如果在理想范围附近，选择最接近理想距离的位置
                        else:
                            min_distance_diff = distance_diff
                            best_move = move
                
                if best_move:
                    action.move = True
                    action.move_target = best_move
                else:
                    action.move = False
            else:
                action.move = False
            
            # 如果在攻击范围内，则攻击
            if nearest_distance <= current_piece.attack_range:
                action.attack = True
                action.attack_context = AttackContext()
                action.attack_context.attacker = current_piece
                action.attack_context.target = target_enemy
                action.attack_context.attackPosition = current_piece.position
            else:
                action.attack = False
            
            # 暂不使用法术
            action.spell = False
            
            return action
        
        return strategy

    @staticmethod
    def get_random_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """随机选择一个初始化策略"""
        import random
        strategies = [
            StrategyFactory.get_aggressive_init_strategy(),
            StrategyFactory.get_defensive_init_strategy()
        ]
        return random.choice(strategies)

    @staticmethod
    def get_random_action_strategy() -> Callable[[Environment], ActionSet]:
        """随机选择一个行动策略"""
        import random
        strategies = [
            StrategyFactory.get_aggressive_action_strategy(),
            StrategyFactory.get_defensive_action_strategy()
        ]
        return random.choice(strategies)

    @staticmethod
    def get_alpha_beta_action_strategy(max_depth: int = 3) -> Callable[[Environment], ActionSet]:
        """获取基于AlphaBeta剪枝的行动策略
        
        Args:
            max_depth: 最大搜索深度
            
        Returns:
            Callable[[Environment], ActionSet]: 策略函数
        """
        def alpha_beta(env: Environment, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[ActionSet]]:
            if depth == 0 or env.is_game_over:
                return get_state_score(env), None
                
            current_piece = env.current_piece
            if maximizing:
                max_eval = float('-inf')
                best_action = None
                
                # 获取所有可能的行动
                legal_moves = get_legal_moves(env)
                attackable_targets = get_attackable_targets(env)
                
                # 获取当前棋子可用的法术
                spells = env.get_available_spells(current_piece)
                
                # 尝试每个可能的行动组合
                for move in [None] + legal_moves:
                    # 如果已经没有行动点，跳过移动
                    if move is not None and current_piece.action_points <= 0:
                        continue
                        
                    for target in [None] + attackable_targets:
                        # 如果已经没有行动点，跳过攻击
                        if target is not None and current_piece.action_points <= 0:
                            continue
                            
                        for spell in [None] + spells:
                            # 如果已经没有行动点或法术位，跳过法术
                            if spell is not None and (current_piece.action_points <= 0 or current_piece.spell_slots <= 0):
                                continue
                                
                            action = ActionSet()
                            next_env = fork_environment(env)
                            remaining_points = current_piece.action_points
                            
                            # 设置移动
                            if move is not None and remaining_points > 0:
                                action.move = True
                                action.move_target = move
                                remaining_points -= 1
                            else:
                                action.move = False
                            
                            # 设置攻击
                            if target is not None and remaining_points > 0:
                                action.attack = True
                                action.attack_context = AttackContext()
                                action.attack_context.attacker = current_piece
                                action.attack_context.target = target
                                remaining_points -= 1
                            else:
                                action.attack = False
                            
                            # 设置法术
                            if spell is not None and remaining_points > 0 and current_piece.spell_slots > 0:
                                action.spell = True
                                action.spell_context = SpellContext()
                                action.spell_context.caster = current_piece
                                action.spell_context.target = target if target else current_piece  # 如果没有目标就施放在自己身上
                                action.spell_context.spell = spell
                                action.spell_context.target_area = Area(
                                    current_piece.position.x,
                                    current_piece.position.y,
                                    2  # 默认范围
                                )
                                remaining_points -= 1
                            else:
                                action.spell = False
                            
                            # 模拟行动
                            next_env.execute_player_action(action)
                            
                            eval, _ = alpha_beta(next_env, depth - 1, alpha, beta, False)
                            if eval > max_eval:
                                max_eval = eval
                                best_action = action
                                
                            alpha = max(alpha, eval)
                            if beta <= alpha:
                                break
                        if beta <= alpha:
                            break
                    if beta <= alpha:
                        break
                            
                return max_eval, best_action
            else:
                min_eval = float('inf')
                best_action = None
                
                # 获取所有可能的行动
                legal_moves = get_legal_moves(env)
                attackable_targets = get_attackable_targets(env)
                
                # 创建基础法术列表
                spells = []
                if current_piece.spell_slots > 0:
                    spells.extend([
                        Spell("Damage", "Damage", 10, False),
                        Spell("Heal", "Heal", 8, False),
                        Spell("Buff", "Buff", 5, False),
                        Spell("Debuff", "Debuff", 3, False)
                    ])
                
                # 尝试每个可能的行动组合
                for move in [None] + legal_moves:
                    # 如果已经没有行动点，跳过移动
                    if move is not None and current_piece.action_points <= 0:
                        continue
                        
                    for target in [None] + attackable_targets:
                        # 如果已经没有行动点，跳过攻击
                        if target is not None and current_piece.action_points <= 0:
                            continue
                            
                        for spell in [None] + spells:
                            # 如果已经没有行动点或法术位，跳过法术
                            if spell is not None and (current_piece.action_points <= 0 or current_piece.spell_slots <= 0):
                                continue
                                
                            action = ActionSet()
                            next_env = fork_environment(env)
                            remaining_points = current_piece.action_points
                            
                            # 设置移动
                            if move is not None and remaining_points > 0:
                                action.move = True
                                action.move_target = move
                                remaining_points -= 1
                            else:
                                action.move = False
                            
                            # 设置攻击
                            if target is not None and remaining_points > 0:
                                action.attack = True
                                action.attack_context = AttackContext()
                                action.attack_context.attacker = current_piece
                                action.attack_context.target = target
                                remaining_points -= 1
                            else:
                                action.attack = False
                            
                            # 设置法术
                            if spell is not None and remaining_points > 0 and current_piece.spell_slots > 0:
                                # 获取法术可选目标
                                spell_targets = env.get_spell_targets(spell, current_piece)
                                if not spell_targets and not spell.is_area_effect:
                                    continue
                                    
                                action.spell = True
                                action.spell_context = SpellContext()
                                action.spell_context.caster = current_piece
                                action.spell_context.spell = spell
                                
                                # 设置目标和范围
                                if spell.is_area_effect:
                                    # 范围法术以当前位置为中心
                                    action.spell_context.target = None
                                    action.spell_context.target_area = Area(
                                        current_piece.position.x,
                                        current_piece.position.y,
                                        spell.area_radius
                                    )
                                else:
                                    # 单体法术选择最佳目标
                                    best_target = None
                                    if spell.effect_type in [SpellEffectType.DAMAGE, SpellEffectType.DEBUFF]:
                                        # 选择生命值最低的敌人
                                        best_target = min(spell_targets, key=lambda p: p.health)
                                    elif spell.effect_type in [SpellEffectType.HEAL, SpellEffectType.BUFF]:
                                        # 选择生命值损失最多的友军
                                        best_target = min(spell_targets, key=lambda p: p.health / p.max_health)
                                    elif spell.effect_type == SpellEffectType.MOVE:
                                        best_target = current_piece
                                        
                                    action.spell_context.target = best_target
                                    action.spell_context.target_area = Area(
                                        best_target.position.x,
                                        best_target.position.y,
                                        0
                                    )
                                    
                                remaining_points -= 1
                            else:
                                action.spell = False
                            
                            # 模拟行动
                            next_env.execute_player_action(action)
                            
                            eval, _ = alpha_beta(next_env, depth - 1, alpha, beta, True)
                            if eval < min_eval:
                                min_eval = eval
                                best_action = action
                                
                            beta = min(beta, eval)
                            if beta <= alpha:
                                break
                        if beta <= alpha:
                            break
                    if beta <= alpha:
                        break
                            
                return min_eval, best_action
        
        def strategy(env: Environment) -> ActionSet:
            _, best_action = alpha_beta(env, max_depth, float('-inf'), float('inf'), True)
            return best_action if best_action is not None else ActionSet()
            
        return strategy
        
  
    def get_mcts_action_strategy(simulation_count: int = 10) -> Callable[[Environment], ActionSet]:
        """获取基于MCTS的行动策略
        
        Args:
            simulation_count: 每个决策点的模拟次数
            
        Returns:
            Callable[[Environment], ActionSet]: 策略函数
        """
        class MCTSNode:
            def __init__(self, env: Environment, parent=None, action: Optional[ActionSet] = None):
                self.env = env
                self.parent = parent
                self.action = action
                self.children = []
                self.visits = 0
                self.value = 0.0
                
            def expand(self):
                """扩展当前节点"""
                current_piece = self.env.current_piece
                legal_moves = get_legal_moves(self.env)
                attackable_targets = get_attackable_targets(self.env)
                
                # 获取当前棋子可用的法术
                spells = self.env.get_available_spells(current_piece)
                
                # print(f"[MCTS] 开始生成动作组合:")
                # print(f"[MCTS] - 可移动位置: {len(legal_moves)}")
                # print(f"[MCTS] - 可攻击目标: {len(attackable_targets)}")
                # print(f"[MCTS] - 可用法术: {len(spells)}")
                # print(f"[MCTS] - 当前行动点: {current_piece.action_points}")

                # 生成所有可能的行动组合
                for move in [None] + legal_moves:
                    # 如果已经没有行动点，跳过移动
                    if move is not None and current_piece.action_points <= 0:
                        if MCTS_VERBOSE: print("[MCTS] 跳过移动：没有足够的行动点")
                        continue
                        
                    for target in [None] + attackable_targets:
                        # 如果已经没有行动点，跳过攻击
                        if target is not None and current_piece.action_points <= 0:
                            if MCTS_VERBOSE: print("[MCTS] 跳过攻击：没有足够的行动点")
                            continue
                            
                        for spell in [None] + spells:
                            # 如果已经没有行动点或法术位，跳过法术
                            if spell is not None and (current_piece.action_points <= 0 or current_piece.spell_slots <= 0):
                                if MCTS_VERBOSE: print("[MCTS] 跳过法术：没有足够的资源")
                                continue
                                
                            action = ActionSet()
                            next_env = fork_environment(self.env)
                            remaining_points = current_piece.action_points
                            has_action = False  # 标记是否有任何动作
                            
                            # 设置移动
                            if move is not None and remaining_points > 0:
                                action.move = True
                                action.move_target = move
                                remaining_points -= 1
                                has_action = True
                                if MCTS_VERBOSE: print(f"[MCTS] 添加移动到 ({move.x}, {move.y})")
                            else:
                                action.move = False
                            
                            # 设置攻击
                            if target is not None and remaining_points > 0:
                                action.attack = True
                                action.attack_context = AttackContext()
                                action.attack_context.attacker = current_piece
                                action.attack_context.target = target
                                remaining_points -= 1
                                has_action = True
                                if MCTS_VERBOSE: print(f"[MCTS] 添加攻击目标 {target.id}")
                            else:
                                action.attack = False
                            
                            # 设置法术
                            if spell is not None and remaining_points > 0 and current_piece.spell_slots > 0:
                                # 获取法术可选目标
                                spell_targets = self.env.get_spell_targets(spell, current_piece)
                                if not spell_targets and not spell.is_area_effect:
                                    if MCTS_VERBOSE: print("[MCTS] 跳过法术：没有有效目标")
                                    continue
                                    
                                has_action = True
                                    
                                action.spell = True
                                action.spell_context = SpellContext()
                                action.spell_context.caster = current_piece
                                action.spell_context.spell = spell
                                if MCTS_VERBOSE: print(f"[MCTS] 添加法术 {spell.name}")
                                
                                # 设置目标和范围
                                if spell.is_area_effect:
                                    # 范围法术以当前位置为中心
                                    action.spell_context.target = None
                                    action.spell_context.target_area = Area(
                                        current_piece.position.x,
                                        current_piece.position.y,
                                        spell.area_radius
                                    )
                                else:
                                    # 单体法术选择最佳目标
                                    best_target = None
                                    if spell.effect_type in [SpellEffectType.DAMAGE, SpellEffectType.DEBUFF]:
                                        # 选择生命值最低的敌人
                                        best_target = min(spell_targets, key=lambda p: p.health)
                                    elif spell.effect_type in [SpellEffectType.HEAL, SpellEffectType.BUFF]:
                                        # 选择生命值损失最多的友军
                                        best_target = min(spell_targets, key=lambda p: p.health / p.max_health)
                                    elif spell.effect_type == SpellEffectType.MOVE:
                                        best_target = current_piece
                                        
                                    action.spell_context.target = best_target
                                    action.spell_context.target_area = Area(
                                        best_target.position.x,
                                        best_target.position.y,
                                        spell.area_radius
                                    )
                                    
                                remaining_points -= 1
                            else:
                                action.spell = False
                            
                            # 如果有行动点但没有执行任何动作，跳过这个组合
                            if current_piece.action_points > 0 and not has_action:
                                if MCTS_VERBOSE: print("[MCTS] 跳过：有行动点但未执行任何动作")
                                continue
                                
                            # 创建子节点并执行完整的步进
                            if MCTS_VERBOSE: print(f"[MCTS] 尝试动作: {action}")
                            step_with_action(next_env, action)
                            child = MCTSNode(next_env, self, action)
                            self.children.append(child)
                            if MCTS_VERBOSE: print(f"[MCTS] 成功添加子节点，当前共有 {len(self.children)} 个子节点")
                        
            def select(self) -> 'MCTSNode':
                """选择最有希望的子节点"""
                if not self.children:
                    return self
                    
                # UCB1公式选择节点
                def ucb1(node: MCTSNode) -> float:
                    if node.visits == 0:
                        return float('inf')
                    return node.value / node.visits + math.sqrt(2 * math.log(self.visits) / node.visits)
                    
                return max(self.children, key=ucb1)
                
            def simulate(self) -> float:
                """模拟到游戏结束或达到最大步数
                
                Returns:
                    float: 1.0 表示当前行动方胜利，-1.0 表示对手胜利，
                          如果达到最大步数，则根据双方棋子血量总和判断胜负
                """
                sim_env = fork_environment(self.env)
                max_steps = 50  # 最大模拟步数
                initial_team = sim_env.current_piece.team  # 记录当前行动方
                
                while not sim_env.is_game_over and max_steps > 0:
                    # 随机选择行动
                    legal_moves = get_legal_moves(sim_env)
                    attackable_targets = get_attackable_targets(sim_env)
                    
                    action = ActionSet()
                    
                    # 随机移动
                    if legal_moves and random.random() < 0.7:
                        action.move = True
                        action.move_target = random.choice(legal_moves)
                    else:
                        action.move = False
                        
                    # 随机攻击
                    if attackable_targets and random.random() < 0.8:
                        action.attack = True
                        action.attack_context = AttackContext()
                        action.attack_context.attacker = sim_env.current_piece
                        action.attack_context.target = random.choice(attackable_targets)
                    else:
                        action.attack = False
                        
                    action.spell = False
                    
                    # 执行模拟动作
                    step_with_action(sim_env, action)
                    max_steps -= 1
                
                # 如果游戏已经结束，直接根据胜负返回结果
                if sim_env.is_game_over:
                    team1_alive = any(p.is_alive for p in sim_env.player1.pieces)
                    team2_alive = any(p.is_alive for p in sim_env.player2.pieces)
                    if team1_alive and not team2_alive:
                        return 1.0 if initial_team == 1 else -1.0
                    elif team2_alive and not team1_alive:
                        return 1.0 if initial_team == 2 else -1.0
                    else:
                        return 0.0  # 平局
                
                # 如果没有执行任何动作，给予惩罚
                if not (action.move or action.attack or action.spell):
                    return -0.5  # 不行动的惩罚值
                
                # 如果达到最大步数，根据双方棋子血量总和判断
                team1_health = sum(p.health for p in sim_env.player1.pieces if p.is_alive)
                team2_health = sum(p.health for p in sim_env.player2.pieces if p.is_alive)
                
                if team1_health > team2_health:
                    return 1.0 if initial_team == 1 else -1.0
                elif team2_health > team1_health:
                    return 1.0 if initial_team == 2 else -1.0
                else:
                    return 0.0  # 血量相等，平局
                
            def backpropagate(self, value: float):
                """反向传播模拟结果"""
                node = self
                while node is not None:
                    node.visits += 1
                    node.value += value
                    node = node.parent
                    value = -value  # 对抗游戏中，父节点的收益是子节点的相反数
        
        def strategy(env: Environment) -> ActionSet:
            root = MCTSNode(env)
            
            # 运行MCTS
            for _ in range(simulation_count):
                node = root
                
                # 选择
                while node.children:
                    node = node.select()
                    
                # 扩展
                if node.visits > 0:
                    node.expand()
                    if node.children:
                        node = random.choice(node.children)
                        
                # 模拟
                value = node.simulate()
                
                # 反向传播
                node.backpropagate(value)
                
            # 选择访问次数最多的子节点对应的行动
            if not root.children:
                if MCTS_VERBOSE:
                    print("\n[MCTS] 警告: 没有生成任何子节点!")
                    print(f"[MCTS] 当前棋子: ID={env.current_piece.id if env.current_piece else None}")
                    print(f"[MCTS] 可移动位置数量: {len(get_legal_moves(env))}")
                    print(f"[MCTS] 可攻击目标数量: {len(get_attackable_targets(env))}")
                    print(f"[MCTS] 可用法术数量: {len(env.get_available_spells())}")
                    print(f"[MCTS] 当前行动点: {env.current_piece.action_points if env.current_piece else 0}")
                    print(f"[MCTS] 当前法术位: {env.current_piece.spell_slots if env.current_piece else 0}")
                return ActionSet()
                
            if MCTS_VERBOSE:
                print(f"\n[MCTS] 找到 {len(root.children)} 个可能的动作")
            best_child = max(root.children, key=lambda c: c.visits)
            if MCTS_VERBOSE:
                print(f"[MCTS] 选择最佳动作: 访问次数={best_child.visits}, 评分={best_child.value}")
                print(f"[MCTS] 动作详情:\n{best_child.action}")
            return best_child.action
        
        return strategy
