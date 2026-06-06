# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from packaging import version


def version_ge(module, target_version):
    return version.parse(module.__version__) >= version.parse(target_version)
