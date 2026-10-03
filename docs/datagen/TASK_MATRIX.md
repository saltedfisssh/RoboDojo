# 上游任务配置工程库存

本表包含此次35类之外的上游配置，供查找代码适配状态；**不能据此判断某任务已经稳定批量生成，或任意layout可用**。逐类已验证范围见 [35类planner能力表](CAPABILITIES.md)，扩大布局见 [操作指南](LAYOUT_GUIDE.md)。

这是工程难度判断；implemented 只表示已有程序，不等于物理任务成功。实际 35 类采集状态见 [检查目录](../../data/datagen_review35/REVIEW.md)。experimental 表示需要独立恢复测试。

| 任务配置 | 对象类型 | 采集程序 | 状态适配 | 难度 | 后续工作 |
| --- | --- | --- | --- | --- | --- |
| align_blocks | Rigid | not_implemented | rigid_candidate | high | Tool grasp and sustained nonprehensile contact; completion predicates |
| arrange_largest_number | Rigid, Geometry | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| arrange_largest_number_random | Rigid, Geometry | not_implemented | rigid_candidate | medium | Asset-specific grasps, goal poses and task ordering |
| build_tower | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| classify_objects | Rigid, Geometry | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| classify_objects_by_language | Rigid, Geometry | not_implemented | rigid_candidate | medium | Asset-specific grasps, goal poses and task ordering |
| cover_blocks | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| deposit_coin | Rigid, Geometry | implemented | rigid | high | Functional grasp/insertion frames and constrained Cartesian/contact control |
| fasten_screws | Rigid, Geometry | implemented | rigid | high | Functional grasp/insertion frames and constrained Cartesian/contact control |
| fill_egg_holder | Rigid, Geometry, Articulation | implemented | articulated_experimental | high | Object joint state, drive targets and task-specific contact skills |
| fill_pen_holder | Rigid | implemented | rigid | medium_high | Coordinated two-arm grasp/handover and obstacle checks |
| fold_clothes | Garment | implemented | particle_dynamic_experimental | very_high | Particle/node state, cloth self-contact and a folding policy |
| fold_clothes_random | Garment | not_implemented | requires_adapter | very_high | Particle/node state, cloth self-contact and a folding policy |
| general_pickup | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| hang_mugs | Rigid, Geometry | implemented | rigid | high | Functional grasp/insertion frames and constrained Cartesian/contact control |
| hang_mugs_random | Rigid, Geometry | not_implemented | rigid_candidate | high | Functional grasp/insertion frames and constrained Cartesian/contact control |
| imitate_sorting_sequence | Rigid, Geometry | implemented | interactive_experimental | high | Support-arm script queues, interaction phase and memory state |
| insert_key | Rigid, Geometry | implemented | rigid | high | Functional grasp/insertion frames and constrained Cartesian/contact control |
| insert_tubes | Rigid, Geometry | implemented | rigid | high | Functional grasp/insertion frames and constrained Cartesian/contact control |
| make_kong | Rigid | implemented | interactive_experimental | high | Support-arm script queues, interaction phase and memory state |
| make_toast | Rigid, Geometry, Articulation | implemented | articulated_experimental | high | Object joint state, drive targets and task-specific contact skills |
| make_toast_random | Rigid, Geometry, Articulation | not_implemented | requires_adapter | high | Object joint state, drive targets and task-specific contact skills |
| match_and_pick_from_conveyor | Rigid, Dynamic | implemented | particle_dynamic_experimental | high | Conveyor body state, clock, spawn schedule and moving-target interception |
| organize_table | Rigid, Geometry | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| pack_objects_into_box | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| pack_objects_into_box_random | Rigid | not_implemented | rigid_candidate | medium | Asset-specific grasps, goal poses and task ordering |
| pick_from_conveyor_by_image | Rigid, Geometry, Dynamic | not_implemented | requires_adapter | high | Conveyor body state, clock, spawn schedule and moving-target interception |
| play_Xylophone | Rigid, Geometry | implemented | rigid | high | Tool grasp and sustained nonprehensile contact; completion predicates |
| play_stacking_toy | Rigid, Geometry | implemented | rigid | high | Functional grasp/insertion frames and constrained Cartesian/contact control |
| play_tic_tac_toe | Rigid, Geometry | implemented | interactive_experimental | high | Support-arm script queues, interaction phase and memory state |
| plug_in_charger | Rigid | implemented | rigid | high | Functional grasp/insertion frames and constrained Cartesian/contact control |
| pour_balls_into_vase | Rigid, Geometry | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| pour_by_language | Rigid, Fluid | not_implemented | requires_adapter | very_high | Particle state and emitter/system parameters; pouring feedback |
| pour_liquid_into_cup | Rigid, Fluid | implemented | particle_dynamic_experimental | very_high | Particle state and emitter/system parameters; pouring feedback |
| pour_liquid_into_cup_random | Rigid, Fluid | not_implemented | requires_adapter | very_high | Particle state and emitter/system parameters; pouring feedback |
| press_by_number | Geometry, Articulation | implemented | articulated_experimental | high | Object joint state, drive targets and task-specific contact skills |
| push_T | Rigid, Geometry | implemented | rigid | high | Tool grasp and sustained nonprehensile contact; completion predicates |
| push_T_random | Rigid, Geometry | not_implemented | rigid_candidate | high | Tool grasp and sustained nonprehensile contact; completion predicates |
| put_bottles_into_dustbin | Rigid, Geometry | implemented | rigid | medium_high | Coordinated two-arm grasp/handover and obstacle checks |
| solve_equation | Rigid, Geometry | not_implemented | rigid_candidate | medium | Asset-specific grasps, goal poses and task ordering |
| sort_nesting_dolls_by_size | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| sort_nesting_dolls_by_size_random | Rigid | not_implemented | rigid_candidate | medium | Asset-specific grasps, goal poses and task ordering |
| stack_blocks | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| stack_blocks_by_language | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| stack_blocks_random | Rigid | not_implemented | rigid_candidate | medium | Asset-specific grasps, goal poses and task ordering |
| stack_bowls | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| stack_bowls_random | Rigid | not_implemented | rigid_candidate | medium | Asset-specific grasps, goal poses and task ordering |
| store_laptop_and_headphones | Rigid, Geometry, Articulation | implemented | articulated_experimental | high | Object joint state, drive targets and task-specific contact skills |
| store_laptop_and_headphones_random | Rigid, Geometry, Articulation | not_implemented | requires_adapter | high | Object joint state, drive targets and task-specific contact skills |
| store_tools_in_toolbox | Rigid, Geometry | not_implemented | rigid_candidate | medium_high | Coordinated two-arm grasp/handover and obstacle checks |
| swap_T | Rigid | implemented | rigid | medium | Asset-specific grasps, goal poses and task ordering |
| swap_blocks | Rigid, Geometry, Articulation | implemented | articulated_experimental | high | Object joint state, drive targets and task-specific contact skills |
| sweep_blocks | Rigid | implemented | rigid | high | Tool grasp and sustained nonprehensile contact; completion predicates |
| sweep_blocks_random | Rigid | not_implemented | rigid_candidate | high | Tool grasp and sustained nonprehensile contact; completion predicates |

数据集的拼写任务缺失上游类，本地 display_letters 是标明来源的重建定义，不在此上游配置表中。
