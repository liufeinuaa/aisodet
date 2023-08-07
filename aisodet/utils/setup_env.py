# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope

"""
自编的的注册自定义的模块到mmengine中

"""

def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in aisodet into the registries.

    Args:
        init_default_scope (bool): Whether initialize the aisodet default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `aisodet`, an aisodet all registries will build modules from aisodet's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa

    import aisodet.datasets  # 我这个工程中只引入自定义的datasets，其他模型通过类似project/的方法来加载我自定义的模型方法
    import aisodet.evaluation  # noqa: F401,F403
    import aisodet.models  # noqa: F401,F403
    import aisodet.visualization  # noqa: F401,F403
    import aisodet.hooks

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('aisodet')
        if never_created:
            DefaultScope.get_instance('aisodet', scope_name='aisodet')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'aisodet':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "aisodet", '
                          '`register_all_modules` will force the current'
                          'default scope to be "aisodet". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'aisodet-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='aisodet')






