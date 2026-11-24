#!/bin/bash
# Copy the harness code to ~/synthlab/harness/ on the Jetson (code only; models stay).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
"$here/tools/jetson.sh" run 'mkdir -p ~/synthlab/harness/models ~/synthlab/harness/runs'
"$here/tools/jetson.sh" push "$here/synthlab/harness/" '/home/jetsonnano/synthlab/harness/' 
