import { Button, Select } from 'antd';
import React, { PropsWithChildren, useCallback, useMemo, useState } from 'react';

import css from './TableBatch.module.scss';

interface Action {
  bulk?: boolean;
  disabled?: boolean;
  label: string;
  value: string;
}

interface Props {
  actions?: Action[];
  ids?: string[];
  onAction?: (action: string) => void;
  onChangeSelectionMode? : () => void;
  onClear?: () => void;
  selectAllMatching?: boolean
  selectedRowCount?: number;
}

const defaultProps = {
  ids: [],
  selectedRowCount: 0,
};

const TableBatch: React.FC<Props> = ({
  actions,
  selectedRowCount,
  selectAllMatching,
  onAction,
  onClear,
  onChangeSelectionMode,
}: PropsWithChildren<Props>) => {
  const [ action, setAction ] = useState<string>();
  const classes = [ css.base ];
  const selectCount = selectedRowCount || 0;

  const selectAllMatchingEnabled = useMemo(() => actions?.some((a) => !!a.bulk), [ actions ]);

  const message = selectAllMatching
    ? 'Apply batch operations to all matching items'
    : `Apply batch operations to ${selectCount}` +
    ` item${selectCount === 1 ? '' : 's'}`;

  if (selectCount > 0) classes.push(css.show);

  const handleAction = useCallback((action?: string) => {
    /*
     * This succession setting of action to an empty string
     * followed by `undefined` is required to guarantee clearing
     * out of the selection value. Using a state `value` prop and
     * setting the state to `undefined` did not work.
     */
    setAction('');
    setTimeout(() => setAction(undefined), 100);

    if (action && onAction) onAction(action);
  }, [ onAction ]);

  const handleClear = useCallback(() => {
    if (onClear) onClear();
  }, [ onClear ]);

  return (
    <div className={classes.join(' ')}>
      <div className={css.container}>
        <div className={css.actions}>
          <Select
            options={actions}
            placeholder="Select an action..."
            value={action}
            onSelect={handleAction}
          />
        </div>
        <div className={css.message}>{message}</div>
        <div className={css.selectMode}>
          {selectAllMatchingEnabled && (
            <Button onClick={onChangeSelectionMode}>
              {selectAllMatching ?
                'Individual Selection'
                : 'Select All Matching'
              }
            </Button>
          )}
        </div>
        {!selectAllMatching && (
          <div className={css.clear}>
            <Button onClick={handleClear}>Clear</Button>
          </div>
        )}
      </div>
    </div>
  );
};

TableBatch.defaultProps = defaultProps;

export default TableBatch;
